import unittest

import torch
import torch.nn.functional as F

from gpt import GPT
from transformer import Transformer

VOCAB = 11


def encoder_decoder():
    return Transformer(8, VOCAB, VOCAB, 16, num_layers=2, n_heads=2, dropout_rate=0.0).eval()


class RawLogitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.src = torch.randint(0, VOCAB, (3, 5))
        self.tgt = torch.randint(0, VOCAB, (3, 4))

    def test_both_families_return_batch_by_length_by_vocabulary_logits(self):
        gpt = GPT(VOCAB, 8, num_layers=1, n_heads=2, dropout_rate=0.0).eval()
        self.assertEqual(gpt(self.tgt).shape, (3, 4, VOCAB))
        self.assertEqual(encoder_decoder()(self.src, self.tgt).shape, (3, 4, VOCAB))

    def test_encoder_decoder_output_is_the_vocabulary_projection(self):
        model = encoder_decoder()
        captured = {}
        hook = model.decoder.fully_connected.register_forward_hook(lambda _m, _i, out: captured.update(out=out))
        self.addCleanup(hook.remove)

        output = model(self.src, self.tgt)

        torch.testing.assert_close(output, captured["out"], rtol=0, atol=0)

    def test_batch_size_one_outputs_are_not_normalized_probabilities(self):
        output = encoder_decoder()(self.src[:1], self.tgt[:1])

        self.assertFalse(torch.all(output == 1.0))
        self.assertTrue((output < 0).any(), "softmax output would be non-negative")
        self.assertFalse(torch.allclose(output.sum(-1), torch.ones(1, 4)))

    def test_cross_entropy_and_gradients_match_a_direct_logit_calculation(self):
        model = encoder_decoder()
        projection = model.decoder.fully_connected
        labels = torch.randint(0, VOCAB, (3, 4)).reshape(-1)

        def loss_and_gradients(direct):
            model.zero_grad()
            hidden = {}
            hook = projection.register_forward_hook(lambda _m, inputs, _out: hidden.update(x=inputs[0]))
            output = model(self.src, self.tgt)
            hook.remove()
            # The reference projects the final hidden states itself, bypassing whatever forward returns.
            logits = F.linear(hidden["x"], projection.weight, projection.bias) if direct else output
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), labels)
            loss.backward()
            return loss.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None]

        model_loss, model_grads = loss_and_gradients(direct=False)
        direct_loss, direct_grads = loss_and_gradients(direct=True)

        torch.testing.assert_close(model_loss, direct_loss)
        self.assertEqual(len(model_grads), len(direct_grads))
        self.assertGreater(len(model_grads), 0)
        for actual, expected in zip(model_grads, direct_grads):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
