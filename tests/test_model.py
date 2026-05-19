import unittest

import torch

from model import build_model


class DynamicImageSizeModelTests(unittest.TestCase):
    def test_autoencoder_preserves_128_and_256_shapes(self) -> None:
        for image_size in (128, 256):
            model = build_model(
                model_type="ae",
                latent_dim=32,
                image_size=image_size,
                use_batch_norm=True,
            )
            inputs = torch.rand(2, 1, image_size, image_size)
            outputs = model(inputs)

            self.assertEqual(tuple(outputs.shape), tuple(inputs.shape))

    def test_vae_preserves_256_shape(self) -> None:
        model = build_model(
            model_type="vae",
            latent_dim=32,
            image_size=256,
            use_batch_norm=False,
        )
        inputs = torch.rand(2, 1, 256, 256)
        outputs = model(inputs)

        self.assertEqual(tuple(outputs["reconstruction"].shape), tuple(inputs.shape))
        self.assertEqual(tuple(outputs["mu"].shape), (2, 32))
        self.assertEqual(tuple(outputs["logvar"].shape), (2, 32))

    def test_rejects_sizes_not_divisible_by_encoder_stride(self) -> None:
        with self.assertRaises(ValueError):
            build_model(model_type="ae", latent_dim=32, image_size=130)


if __name__ == "__main__":
    unittest.main()
