import warnings
from typing import Dict, List, Tuple, Iterable, Any, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell, LSTM

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.common.lazy import Lazy
from allennlp.training.metrics import BLEU

from core.comp.nn.fusion.diff_siso_fusions.diff_siso_fusion import DiffSeqinSeqoutFusion
from utils import GlobalLogger as mylogger


@Model.register("hybrid_cc_simple_seq2seq")
class HybridCcSimpleSeq2Seq(Model):
    """
     Implicit interaction based code change seq2seq model.
     This model mainly uses the code of 'SimpleSeq2Seq' model in AllenNLp_models and
     a few changes have been applied to adapt it to code change task:

     1. Add code change fusion module and its forward logic, which uses encoder to encode
        both added code and removed code and then use fusion to merge these two representations
        to produce final encoder output.
     2. Modify the logic of aligning dimension between encoder and decoder, now encoder output dimension
        is equal to fusion's output dimension.
     3. Use 'target_namespace' in params as key to fetch target tokens instead of
        default 'tokens';
     4. Add support for opearation sequence embedding and fusion.
     5. Add diff as an additional input of 'forward'.

     # ------------------------------------------------------------------------------------------
    Following are raw note:

    This `SimpleSeq2Seq` class is a `Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.
    # Parameters
    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_embedder : `TextFieldEmbedder`, required
        Embedder for source side sequences
    encoder : `Seq2SeqEncoder`, required
        The encoder of the "encoder/decoder" model
    beam_search : `BeamSearch`, optional (default = `Lazy(BeamSearch)`)
        This is used to during inference to select the tokens of the decoded output sequence.
    target_namespace : `str`, optional (default = `'tokens'`)
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : `int`, optional (default = `'source_embedding_dim'`)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    target_pretrain_file : `str`, optional (default = `None`)
        Path to target pretrain embedding files
    target_decoder_layers : `int`, optional (default = `1`)
        Nums of layer for decoder
    attention : `Attention`, optional (default = `None`)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    scheduled_sampling_ratio : `float`, optional (default = `0.`)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015](https://arxiv.org/abs/1506.03099).
    use_bleu : `bool`, optional (default = `True`)
        If True, the BLEU metric will be calculated during validation.
    ngram_weights : `Iterable[float]`, optional (default = `(0.25, 0.25, 0.25, 0.25)`)
        Weights to assign to scores for each ngram size.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        source_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        diff_siso_fusion: DiffSeqinSeqoutFusion = None,
        op_embedder: TextFieldEmbedder = None,
        beam_search: Lazy[BeamSearch] = Lazy(BeamSearch),
        attention: Attention = None,
        target_namespace: str = "tokens",
        target_embedding_dim: int = None,
        scheduled_sampling_ratio: float = 0.0,
        using_decay_sampling: bool = False,
        use_bleu: bool = True,
        bleu_ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
        target_pretrain_file: str = None,
        target_decoder_layers: int = 1,
        decoder_input_dim: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(vocab)
        # * added unique attributes used in cc task
        self.fusion = diff_siso_fusion
        self.op_embedder = op_embedder

        self._target_namespace = target_namespace
        self._target_decoder_layers = target_decoder_layers
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._using_decay_sampling = using_decay_sampling
        self._decay_sampling_p = None

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(
            START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(
            END_SYMBOL, self._target_namespace)

        if use_bleu:
            pad_index = self.vocab.get_token_index(
                self.vocab._padding_token, self._target_namespace
            )
            self._bleu = BLEU(
                bleu_ngram_weights,
                exclude_indices={pad_index,
                                 self._end_index, self._start_index},
            )
        else:
            self._bleu = None

        # At prediction time, we'll use a beam search to find the best target sequence.
        # For backwards compatibility, check if beam_size or max_decoding_steps were passed in as
        # kwargs. If so, update the BeamSearch object before constructing and raise a DeprecationWarning
        deprecation_warning = (
            "The parameter {} has been deprecated."
            " Provide this parameter as argument to beam_search instead."
        )
        beam_search_extras = {}
        if "beam_size" in kwargs:
            beam_search_extras["beam_size"] = kwargs["beam_size"]
            warnings.warn(deprecation_warning.format(
                "beam_size"), DeprecationWarning)
        if "max_decoding_steps" in kwargs:
            beam_search_extras["max_steps"] = kwargs["max_decoding_steps"]
            warnings.warn(deprecation_warning.format(
                "max_decoding_steps"), DeprecationWarning)
        self._beam_search = beam_search.construct(
            end_index=self._end_index, vocab=self.vocab, **beam_search_extras
        )

        # Dense embedding of source vocab tokens.
        self._source_embedder = source_embedder

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = encoder

        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        # Attention mechanism applied to the encoder output for each step.
        self._attention = attention

        # Dense embedding of vocab words in the target space.
        target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()
        if not target_pretrain_file:
            self._target_embedder = Embedding(
                num_embeddings=num_classes, embedding_dim=target_embedding_dim
            )
        else:
            self._target_embedder = Embedding(
                embedding_dim=target_embedding_dim,
                pretrained_file=target_pretrain_file,
                vocab_namespace=self._target_namespace,
                vocab=self.vocab,
            )

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        # **************************************************************************
        # NOTE: encoder output dimension depends on output dim of fusion, not encoder
        # self._encoder.get_output_dim()
        self._encoder_output_dim = self.fusion.get_output_dim()
        # **************************************************************************
        self._decoder_output_dim = self._encoder_output_dim

        # Manually set input dim of decoder
        if decoder_input_dim is not None:
            self._decoder_input_dim = decoder_input_dim
        else:
            if self._attention:
                # If using attention, a weighted average over encoder outputs will be concatenated
                # to the previous target embedding to form the input to the decoder at each
                # time step.
                self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim
            else:
                # Otherwise, the input to the decoder is just the previous target embedding.
                self._decoder_input_dim = target_embedding_dim

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # TODO (pradeep): Do not hardcode decoder cell type.
        if self._target_decoder_layers > 1:
            self._decoder_cell = LSTM(
                self._decoder_input_dim,
                self._decoder_output_dim,
                self._target_decoder_layers,
            )
        else:
            self._decoder_cell = LSTMCell(
                self._decoder_input_dim, self._decoder_output_dim)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(
            self._decoder_output_dim, num_classes)

    def set_decay_sampling_p(self, p):
        self._decay_sampling_p = p

    def take_step(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.
        # Parameters
        last_predictions : `torch.Tensor`
            A tensor of shape `(group_size,)`, which gives the indices of the predictions
            during the last time step.
        state : `Dict[str, torch.Tensor]`
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape `(group_size, *)`, where `*` can be any other number
            of dimensions.
        step : `int`
            The time step in beam search decoding.

        # Returns
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of `(log_probabilities, updated_state)`, where `log_probabilities`
            is a tensor of shape `(group_size, num_classes)` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while `updated_state` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.
        Notes
        -----
            We treat the inputs as a batch, even though `group_size` is not necessarily
            equal to `batch_size`, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(
            last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    @overrides(check_signature=False)
    def forward(
            self,  # type: ignore
            diff: TextFieldTensors,
            diff_add: TextFieldTensors,
            diff_del: TextFieldTensors,
            diff_op: TextFieldTensors = None,
            msg: TextFieldTensors = None,
            add_op_mask: Optional[torch.Tensor] = None,
            del_op_mask: Optional[torch.Tensor] = None,
            add_line_idx: Optional[torch.Tensor] = None,
            del_line_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        diff_state = self._encode(diff)
        add_state = self._encode(diff_add)
        del_state = self._encode(diff_del)
        if diff_op is not None:
            assert self.op_embedder is not None
            op_state = self.op_embedder(diff_op)
        else:
            op_state = None

        # code change feature fusion, this is the main difference between this
        # seq2seq and simple_seq2seq
        state = self.fusion(diff_state, add_state, del_state, op_state,
                            add_op_mask, del_op_mask, add_line_idx, del_line_idx)
        if msg:
            state = self._init_decoder_state(state)
            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict = self._forward_loop(state, msg)
        else:
            output_dict = {}

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if msg and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                self._bleu(best_predictions,
                           msg[self._target_namespace]["tokens"])

        return output_dict

    @overrides(check_signature=False)
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize predictions.
        This method overrides `Model.make_output_human_readable`, which gets called after `Model.forward`, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the `forward` method.
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called `predicted_tokens` to the `output_dict`.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for top_k_predictions in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # we want top-k results.
            if len(top_k_predictions.shape) == 1:
                top_k_predictions = [top_k_predictions]

            batch_predicted_tokens = []
            for indices in top_k_predictions:
                indices = list(indices)
                # Collect indices till the first end_symbol
                if self._end_index in indices:
                    indices = indices[: indices.index(self._end_index)]
                predicted_tokens = [
                    self.vocab.get_token_from_index(
                        x, namespace=self._target_namespace)
                    for x in indices
                ]
                batch_predicted_tokens.append(predicted_tokens)

            all_predicted_tokens.append(batch_predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, encoder_output_dim)
        # Get initialized state from upstream output states
        if 'decoder_init_state' in state:
            final_encoder_output = state['decoder_init_state']
        # Else initialize the decoder hidden state with the final output of the encoder.
        else:
            final_encoder_output = util.get_final_encoder_states(
                state["encoder_outputs"],
                state["source_mask"],
                self._encoder.is_bidirectional(),
            )
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"].new_zeros(
            batch_size, self._decoder_output_dim
        )
        if self._target_decoder_layers > 1:
            # shape: (num_layers, batch_size, decoder_output_dim)
            state["decoder_hidden"] = (
                state["decoder_hidden"].unsqueeze(0).repeat(
                    self._target_decoder_layers, 1, 1)
            )

            # shape: (num_layers, batch_size, decoder_output_dim)
            state["decoder_context"] = (
                state["decoder_context"].unsqueeze(0).repeat(
                    self._target_decoder_layers, 1, 1)
            )

        return state

    def _forward_loop(
        self, state: Dict[str, torch.Tensor], target_tokens: TextFieldTensors = None
    ) -> Dict[str, torch.Tensor]:
        """
        This method is almost the same as '_forward_loop' method in 'SimpleSeq2Seq' model,
        except for using target namespace to fetch target sequence tokens:
        `targets = target_tokens[self.target_namespace]["tokens"]`

        Following are raw notes:

        Make forward pass during training or do greedy search during prediction.
        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]

        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens[self._target_namespace]["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._beam_search.max_steps

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full(
            (batch_size,), fill_value=self._start_index, dtype=torch.long
        )

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif self.training and self._using_decay_sampling:
                p = torch.rand(1).item()
                if p > self._decay_sampling_p:
                    # Use predictions at training time with prob 1-p
                    input_choices = last_predictions
                else:
                    # Use ground-truth at training time with prob p
                    input_choices = targets[:, timestep]
            elif not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(
                input_choices, state)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index, dtype=torch.long
        )

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step
        )

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def _prepare_output_projections(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.
        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (num_layers, group_size, decov9der_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (num_layers, group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        if self._attention:
            attention_func = self._prepare_attended_input
            # shape: (group_size, encoder_output_dim)
            if self._target_decoder_layers > 1:
                attended_input = attention_func(
                    decoder_hidden[0], encoder_outputs, source_mask
                )
            else:
                attended_input = attention_func(
                    decoder_hidden, encoder_outputs, source_mask
                )
            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = embedded_input

        # Append fused context if existed
        if 'fused_cat_context' in state:
            decoder_input = torch.cat(
                (decoder_input, state['fused_cat_context']), -1)

        if self._target_decoder_layers > 1:
            # shape: (1, batch_size, target_embedding_dim)
            decoder_input = decoder_input.unsqueeze(0)

            # shape (decoder_hidden): (num_layers, batch_size, decoder_output_dim)
            # shape (decoder_context): (num_layers, batch_size, decoder_output_dim)
            _, (decoder_hidden, decoder_context) = self._decoder_cell(
                decoder_input.float(), (decoder_hidden.float(), decoder_context.float())
            )
        else:
            # shape (decoder_hidden): (batch_size, decoder_output_dim)
            # shape (decoder_context): (batch_size, decoder_output_dim)
            decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input.float(), (decoder_hidden.float(), decoder_context.float())
            )

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # shape: (group_size, num_classes)
        if self._target_decoder_layers > 1:
            output_projections = self._output_projection_layer(
                decoder_hidden[-1])
        else:
            output_projections = self._output_projection_layer(decoder_hidden)
        return output_projections, state

    def _prepare_attended_input(
        self,
        decoder_hidden_state: torch.LongTensor = None,
        encoder_outputs: torch.LongTensor = None,
        encoder_outputs_mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(
            decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    @staticmethod
    def _get_loss(
        logits: torch.LongTensor,
        targets: torch.LongTensor,
        target_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Compute loss.
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.
        The length of `targets` is expected to be greater than that of `logits` because the
        decoder does not need to compute the output corresponding to the last timestep of
        `targets`. This method aligns the inputs appropriately to compute the loss.
        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    @overrides(check_signature=False)
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics

    default_predictor = "seq2seq"
