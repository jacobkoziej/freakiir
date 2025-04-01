# SPDX-License-Identifier: MPL-2.0
#
# layer_test.py -- model layer tests
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import pytest
import torch
import torch.nn as nn

from pytest import FixtureRequest
from torch import (
    Generator,
    Tensor,
    pi,
)

from freakiir.layer import (
    ComplexToReal,
    Mlp,
    MlpConfig,
    RealToComplex,
    ReflectIntoComplexUnitCircle,
)
from freakiir.pdf import uniform


class TestComplexToReal:
    @pytest.fixture(autouse=True)
    def layer(self) -> None:
        self.layer = ComplexToReal()

    @pytest.mark.parametrize(
        "z, r",
        [
            (
                torch.tensor(1.0 + 1.0j),
                torch.tensor([1.0, 1.0]),
            ),
            (
                torch.tensor([1.0 + 1.0j]),
                torch.tensor([[1.0, 1.0]]),
            ),
            (
                torch.tensor(
                    [
                        [1.0 + 1.0j, 2.0 - 2.0j],
                        [3.0 - 3.0j, 4.0 + 4.0j],
                    ]
                ),
                torch.tensor(
                    [
                        [
                            [1.0, 1.0],
                            [2.0, -2.0],
                        ],
                        [
                            [3.0, -3.0],
                            [4.0, 4.0],
                        ],
                    ]
                ),
            ),
        ],
    )
    def test_forward(self, z: Tensor, r: Tensor) -> None:
        assert torch.allclose(self.layer(z), r)


class TestMlpConfig:
    @pytest.fixture(params=[-1, 0, 1])
    def in_features(self, request: FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[-1, 0, 1])
    def hidden_features(self, request: FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[-1, 0, 1])
    def out_features(self, request: FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[-1, 0, 1])
    def hidden_layers(self, request: FixtureRequest) -> int:
        return request.param

    def test_post_init_assert(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        hidden_layers: int,
    ) -> None:
        args = {
            "in_features": in_features,
            "hidden_features": hidden_features,
            "out_features": out_features,
            "hidden_layers": hidden_layers,
        }

        success = True

        for val in args.values():
            success &= val > 0

        if success:
            _ = MlpConfig(**args)
            return

        with pytest.raises(AssertionError):
            _ = MlpConfig(**args)


class TestMlp:
    @pytest.fixture(params=[2, 8])
    def in_features(self, request: FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[4, 3])
    def hidden_features(self, request: FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[8, 2])
    def out_features(self, request: FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[1, 2])
    def hidden_layers(self, request: FixtureRequest) -> int:
        return request.param

    @pytest.fixture(autouse=True)
    def mlp(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        hidden_layers: int,
    ) -> None:
        config = MlpConfig(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            hidden_layers=hidden_layers,
        )

        self.mlp = Mlp(config)

    def test_init_features(self) -> None:
        mlp = self.mlp
        config = mlp.config

        in_features = config.in_features
        hidden_features = config.hidden_features
        out_features = config.out_features

        def get_linear(x: nn.Sequential) -> nn.Linear:
            return x[0]

        in_linear = get_linear(mlp.in_layer)

        assert in_linear.in_features == in_features
        assert in_linear.out_features == hidden_features

        assert len(mlp.hidden_layers) == config.hidden_layers

        for hidden_layer in mlp.hidden_layers:
            hidden_linear = get_linear(hidden_layer)

            assert hidden_linear.in_features == hidden_features
            assert hidden_linear.out_features == hidden_features

        out_linear = get_linear(mlp.out_layer)

        assert out_linear.in_features == hidden_features
        assert out_linear.out_features == out_features

    def test_forward(self) -> None:
        mlp = self.mlp
        config = mlp.config

        x = torch.rand((config.in_features,))

        x = mlp.forward(x)

        assert x.shape == (config.out_features,)


class TestRealToComplex:
    @pytest.fixture(autouse=True)
    def layer(self) -> None:
        self.layer = RealToComplex()

    @pytest.mark.parametrize(
        "r, z",
        [
            (
                torch.tensor([1.0, 1.0]),
                torch.tensor(1.0 + 1.0j),
            ),
            (
                torch.tensor([[1.0, 1.0]]),
                torch.tensor([1.0 + 1.0j]),
            ),
            (
                torch.tensor(
                    [
                        [1.0, 1.0, 2.0, -2.0],
                        [3.0, -3.0, 4.0, 4.0],
                    ]
                ),
                torch.tensor(
                    [
                        [1.0 + 1.0j, 2.0 - 2.0j],
                        [3.0 - 3.0j, 4.0 + 4.0j],
                    ]
                ),
            ),
        ],
    )
    def test_forward(self, r: Tensor, z: Tensor) -> None:
        assert torch.allclose(self.layer(r), z)


class TestReflectIntoComplexUnitCircle:
    def test_forward(
        self,
        generator: Generator,
        samples: int,
    ) -> None:
        pdf = uniform(
            r_a=0,
            r_b=2,
            theta_a=0,
            theta_b=pi,
            generator=generator,
        )

        samples = pdf(samples)

        layer = ReflectIntoComplexUnitCircle()

        inside = layer.forward(samples)

        assert torch.all(inside.abs() <= 1)
