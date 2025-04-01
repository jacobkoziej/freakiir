# SPDX-License-Identifier: MPL-2.0
#
# dataset.py -- datasets
# Copyright (C) 2025  Jacob Koziej <jacobkoziej@gmail.com>

import builtins

import numpy as np
import torch

from dataclasses import dataclass
from ftplib import FTP
from pathlib import Path
from typing import (
    Final,
    Optional,
)
from zipfile import ZipFile

from einops import rearrange
from scipy.io import loadmat
from torch import Tensor
from torch.utils.data import Dataset

from .dsp import (
    construct_sections,
    freqz_zpk,
    order_sections,
)
from .model import (
    ModelInput,
    ModelOutput,
)
from .path import (
    datasets_directory,
    file_hash,
)
from .pdf import Pdf


@dataclass(frozen=True, kw_only=True)
class ListenHrtfOutput(ModelInput):
    elevation: Tensor
    azimuth: Tensor
    content: Tensor
    content_type: list[str]
    sampling_hz: Tensor


class ListenHrtf(Dataset):
    _HASHES: Final[dict[int, str]] = {
        1002: "c5b3131990905678efb07785d9cba53b3a45876af406f9075ad1c0130bdc1dae6eec77f4743b68ef754c7393ba7080c7bbd8a580452629cd3fd5be88e27ca631",
        1003: "955646bb916b8f822db893d33993bdb2f3f71c3e0be21295a1f5daac52631e85e07683a536e4fcd71132270d4453be0d09bb68bdc67e801292e0bb0328403a53",
        1004: "03f3988755c898175902db3d1c33ab4b4c7698a26167d3f61d32e5ccc13d520ed8f9840ff3f371b1e4aee36771d7aa1770abec22a08c8d488f3b355d3fd3d656",
        1005: "c261967939f415742fb5f244ec0017d67a5e86d25d5b88229f8fc868825abe05f7d49759c8b9cb276de40c29426a3e197b99255ebc42b9ec6110ccb93939113b",
        1006: "16fdb4089bd35190090c9d9026ce437c36995160f5a726aee9ea96fe4e947ab0cd630ab67eb14c3d9ae6e5e0a37472d027eafbed3d69c56625a0afadd8b8bfa6",
        1007: "82375f1b38275b4accc70ae6642024d323d20aaa18b23511726c58fd6a01de7821304b49b4f0f5e6371bda8cf1f9f15143b000ae5eb479c90f2c7d67c52a6805",
        1008: "cf492c49ab18cd3589c6cf0559eb9ee5c2984681a0aedc2d2450e7dd3d3600bd74abcb4066e775e742207a3643df7e70523daa6233350adb83131e878b8b4153",
        1009: "3ad124299b3cdbd7e31c46768dad39333454052925169e0755563c49456dd76c6f6334bed31339df3b126b2c2edf168e9c31cd937aaee2a90f58172ec129d6e0",
        1012: "d1823d37e0e412e846a15b5bad56bb2f263e99edc8fbb5719008f58f18207168cd46e8b8c07ccf3cdaa8971a2b288c5acadac382613441a400b0d145179cae4a",
        1013: "4f8381581958e788f47eaf0930eef6bfb432b7c07e2d0fe4e28bc86274acfdf746b0f09deb35431c42a16fff8b9fedda74a15ccb42b56c535cb94d5f64643399",
        1014: "50c34285207e481314a05742c2a09a8135b7b38cc0cf61934f46b9b4a5bf5fc5db2774110449047016a5dcd57e9d6818b7abafdfd20282ceb69c0032e48a506d",
        1015: "9b324af8595b76ce723af4e40deb3869cea0e19d4f08af11f1ab868ac8f1c1850d1480fa8990cdfa2650547411b618f116aabefe357f7237b3d7ae99388c005c",
        1016: "eb2fcbea1b17d60c2f467bfe351d25ee5e84cee9069f73e0c0629c481ccfe00c86efcc1d7dc2616631ee0a1de50ba3b4e583267b58f07acf0cbdbcc0a79763cc",
        1017: "ff52da58dc800a09c8d4cfae06d5a1d80cef9e0fb0b13dd131e6959c70483313ccafe4f1da91536650d5d33d578dda4498f2ade83cb3de12092ea095ce451b79",
        1018: "2c8555bd9ca62c32eecf588138784b6dae79fb4a9bbc264d33a262b03c32954d28502d16b6daae2caef403c4b904403ba95277018b5e4cc149ea9817bc4a9014",
        1020: "ab9921361d4e4c36aa2fb9cdf1f371c9f585d278bf32c844f67042d90b2c96ebac012d6056cae2f774d837441d5e668a20bdd03b857b7700c06c6df400c6a397",
        1021: "07aa3c1cbea0e155089592ce4f02074c78e14c062a65bc0ba327f4da2c08b0baf4747f6ada1bad7fe64ac2f233830899cdd6093521ccc547d09600010e7e712d",
        1022: "a69fbd4a0f98b7cf112300ebc690dbc46d5026db291126cca2b43fcd52b5914825eaaf1298fac4fc01fd3d8f0621961fdde5c7cdfc2444a081c65e429a6a8adb",
        1023: "73fb6d1687f2334ce542b63fffd45a1381a2ad791a487239408b808f8bd341b70aca99acf018fed4528d49d60855ef5078da0f30219b0cde0ae74911c24682b2",
        1025: "ed321e50bfe39510ea7841198e7e9ee5b83cb37032a92ab67f2b9734a338c89290fca14ef567d6818ccd4415f4f287e9e3f74feeb5f1ade031931656d4f2eaa7",
        1026: "c7fc575c3bb42080fc5d4ef01b02ecc006c2669d83d1a5168c4e9aa75b4dadb012272c648424bb66bfebbaa97a5e992b51b7d12f15b0154da0ae26f3ce383049",
        1028: "50c59e728d6bc606b8c7339f3db3ef499b5c19061bbb2571b560a51a31c29454fddd6e1ba788d6782c7d48aedbdfc58f1986dfcb30ed05e04f3b89bdf2d31ec7",
        1029: "d6612ea14685f167fff00adb1f9031e17c3112b5860030b375c4624bf18f018fcfb6cf8b467bcbf4a0d4463a1a3c0bc85dde0c810535b0dc78f736c96daadeee",
        1030: "01aec8a6ab0ed2589b9eec7ce478d92c95622f07dcdd82de59de2d4d2bc4460cc00f0dd4ff4679a4f739c4ecbac9df578cecd482a26402fb7ba342ecee1352fe",
        1031: "3cf7feae7db5cda5db8daa2eec58ae615c18262bc3eb3b7e7c95bd366045c82c2ff126631d1280d4c9a154c92fab03195a2546071d897df08508eff84c0dce7c",
        1032: "7f84952e43b717b82c2647eea5858307c47c1da356a8f84989ce6eb3a820f3bc6274034fe77643d83861bb670a54803fca981c259d5fa57b776c095c770391f4",
        1033: "af5809d93d1470afc3814df34a58364316b7f382658e1253304927084ffcca464bb5f4e6ae48cf41ff94042af4379ca1f023c2ec690c0fa2e92356c2fb80b9c6",
        1034: "7933b72e119ea4a26a933e16774a13a2e367035b4c1f749e1d5440605369ff23bc1b60856204d18bd3ea810bd0f1710831bfea15dba4cb2094541d15994f6b52",
        1037: "88246d956f62d5eac24e2170f42a2388c2677209c153311ab5106393bf20698fc0e9dd1dea242ff068f0baef1d37cbb28ede8ca082ecb486b53b0ea6807eefb6",
        1038: "854efe9c98c09375b283b81727395ba3efd6913d7786c4a6d319bab1b4e675e3bfb4bf4caf717451aca594a681e4b13776d3a9c8f405768c72044e4c29e9ff71",
        1039: "507a33fe93f76e7160d2bc3fd6608772971f271a7e72a5f940eab3d35e5890751568b197ded1839168d50b2cb24cccab7e0a20593871815eea4d2833efe67b05",
        1040: "a36b10333ee55671886a51c596b97763bbcf86b7468506a800f2326c7d3a7fef07b78ba691e5baa81b2f15b684b59a15f3ff925774bea10d6641c7d00b5eb413",
        1041: "8047613a710cb88941dab5cc64dad706b1ca0ad6374150fb4850a702293ccb7470be5994c4e3181e7f231b3cef5addad38033650cffa54e8ba00f493dddde21b",
        1042: "7cd8561b3c4d774dcdbc8cbe0efb56655221e45189f8d508067dd1c75df9d96321c27759c37318b01a65efae0c0a6dca51288718821027b32d226da9918a541d",
        1043: "06b55642535e77017a37669557b4731310722312df30d0f59516f533c91ccfa0ad06bc7415ed69c6b7343062801278e5bc32caba248a4d420e990e953762b4b5",
        1044: "45f88cdfc4232ba35f2c9c14da283ce24a0f4c2715e86e1c726e02f27af4c3e356cbbb41ae7a9bb93c5c0053326b4ff729d1c0ac6a84102d771d566d6680fff1",
        1045: "84c320dfb80dc3d841cfa55f1803d16ab2f7088647daf0c1ca76d832247f2d999c26e8586fff5517671502240a8a6af7a24ff5f8f6dc5b98b1ed2ee7e175ad4d",
        1046: "0e36eafa4a53be731cbf3766afac5f059b5da9dea4d34b22bbfd9d43f51a63e3c196f5576b8a59e646f5196c67232c0f5a130cbbc1187a7f6cd8af1c33aefd48",
        1047: "71191ecce2ac8e0b72b4e81a8b92030007e3c398fa5771eb3ddba35704af94035ce93aec99a1086f58a1f44812e6ba8d01f7db8f510a497fc3228f56d951f064",
        1048: "ef84ece8b5b1355af8d545b33d384c591e7db9c123c13845541aa0cd3b6766c7df888b95d5fa0f622609de00b0b6ff36afb68a3cb43c6abfbd5085dc4769f9dd",
        1049: "01a50fb8dc67ed2fd7b17beca232e2a3a842530c853540783481cf8ecf87214662f50dc40ac4d5445a56331fdc8c8a903ac8b0faa06128cd1f3d218c8b7e3c72",
        1050: "90d0234110336ad8ffc2139f4555b3b8deb7e2ec8523bfd0f817551489daad0f562b2b24615faadf4d70fa8f1877b570bcd36d7d24fb434d03e9db287d8e0080",
        1051: "104ea6b270ed428e6eb53babca9a2fa53003e212c814dad5176efff8d357ddc0f76855183f7a00e8f43d10105f4d8ad22ba9ba018eba9193edd0f51f59437350",
        1052: "5388f9bc27a496c8995d47c089d13e9e01e5b68564d828137970bafe26eb1dbac34ee0491d8a274782b2f777420aa676e1adc70a783c009665c03f244afaadb2",
        1053: "7740e8cd291486d02781f5d012dfb649e385f222027fd41675db28eea5753b6f59814e5dfb3ad2ec40a3afc3b0823e16d2acd8856b1aa85f04dafe845177b73e",
        1054: "e6cd46f509bea5190f75b5dace8fb004710c78d9130429e11e70d31f9f03f90e5b5417016651115f33a6e4b4da1c13d917f47b1a6a03ca9a99111c092702a50f",
        1055: "cb4382d1a22c327c53c86867868eb685f604aef3b9ad55aac76e4a096551d25e25dcca6b3a0e6bb9235d2d72c52ad618b48dab2d83c0bbf05f68a8b54ab2ca3d",
        1056: "fecd22c923d0cec4e9f2e3eae0cb0eaac1857e893c12611bbf8597358cac8cbad6f4eee78b9dd952b209522e9e6e7a2105dcef317bb7da715d6cd0582a4f6fc2",
        1057: "b760313dd6710dc900ef0d97e992c6552f9108397d4f421c53ad9cdc51d517a64d5c362072f02651c22c41b7420be3d21a1e0866cb2ee8591eee7eaf8b3a4b75",
        1058: "23c75e67d8515e1d8fa9794bf392e19fc08f8b6dda5842538d4efc71a3690cea1efe5cfc31a3847f39347a8569a3b66c261cf5468a58f3144a46d861efcd02a0",
        1059: "af99d7f220d9e4df264539cbb793fe160b61382454ee5fc792d6cb17e32f94498e3b694b4e298f307e83060a9a5b5d3093836dfec9355307dcfc655fadfc55e7",
    }

    def __init__(
        self,
        *,
        decompress: bool = True,
        download: bool = True,
        dtype: Optional[torch.dtype] = None,
        root: Optional[Path | str] = None,
    ) -> None:
        if isinstance(root, builtins.str):
            root = Path(root)

        if root is None:
            root = datasets_directory() / "listen-hrtf"

        root.mkdir(parents=True, exist_ok=True)

        self.root = root

        missing = self._check_for_missing_hashes()

        assert download or (not missing)

        if download:
            self._download_missing_hashes(missing)

        self._decompress_dataset(decompress, bool(missing))

        if dtype is None:
            dtype = torch.get_default_dtype()

        self._parse_data(dtype)

    def _check_for_missing_hashes(self) -> list[int]:
        missing = []

        for subject, hash in self._HASHES.items():
            path = self.root / Path(f"IRC_{subject}.zip")

            if not path.is_file():
                missing.append(subject)
                continue

            if file_hash(path) != hash:
                missing.append(subject)

        return missing

    def _decompress_dataset(self, decompress: bool, force: bool) -> None:
        freaked = self.root / ".freaked"

        assert decompress or freaked.is_file()

        if (not force) and (freaked.is_file()):
            return

        for subject in self._HASHES.keys():
            path = self.root / Path(f"IRC_{subject}.zip")

            with ZipFile(path) as zf:
                zf.extractall(self.root)

        freaked.touch()

    def _download_missing_hashes(self, missing: list[int]) -> None:
        ftp = FTP("ftp.ircam.fr")
        ftp.login()
        ftp.cwd("pub/IRCAM/equipes/salles/listen/archive/SUBJECTS")

        for subject in missing:
            subject = Path(f"IRC_{subject}.zip")

            path = self.root / subject

            with open(path, "wb") as fp:
                ftp.retrbinary(f"RETR {str(subject)}", fp.write)

        ftp.quit()

        assert not self._check_for_missing_hashes()

    def _parse_data(self, dtype: torch.dtype) -> None:
        fmt = "COMPENSATED"
        tag = "C"
        struct = "_eq_hrir_S"

        root = self.root / fmt / "MAT" / "HRIR"

        data = []

        for subject in self._HASHES.keys():
            path = root / f"IRC_{subject}_{tag}_HRIR.mat"

            mat = loadmat(path, simplify_cells=True)

            data += [mat["l" + struct], mat["r" + struct]]

        data = {key: [data[key] for data in data] for key in data[0]}

        self.content_type = "FIR"
        self.sampling_hz = 44100

        elevation = torch.tensor(np.array(data["elev_v"]))
        azimuth = torch.tensor(np.array(data["azim_v"])).to(torch.int16)
        content = torch.tensor(np.array(data["content_m"]), dtype=dtype)

        def flatten_position(x: Tensor) -> Tensor:
            return rearrange(
                x,
                "subject position ... -> (subject position) ...",
            )

        self.elevation = flatten_position(elevation)
        self.azimuth = flatten_position(azimuth)
        self.content = flatten_position(content)

        N = self.content.shape[-1]

        self.h = torch.fft.fft(self.content, n=2 * N)[..., :N]

        # our interval is right-open
        end = torch.pi
        end -= end / N

        self.w = torch.linspace(0, end, N)

    def __len__(self) -> int:
        return len(self.h)

    def __getitem__(
        self,
        item: int | slice,
    ) -> ListenHrtfOutput:
        return ListenHrtfOutput(
            w=self.w,
            h=self.h[item],
            elevation=self.elevation[item],
            azimuth=self.azimuth[item],
            content=self.content[item],
            content_type=self.content_type,
            sampling_hz=self.sampling_hz,
        )


@dataclass(frozen=True)
class RandomFilterDatasetConfig:
    sections: int
    pdf_z: Pdf
    pdf_p: Pdf

    all_pass: bool = True
    batch_count: int = 1024
    batch_size: int = 16
    dft_bins: int = 512
    down_order: bool = False
    whole_dft: bool = False

    def __post_init__(self) -> None:
        assert self.sections > 0

        assert self.batch_count > 0
        assert self.batch_size > 0
        assert self.dft_bins > 0


@dataclass(frozen=True, kw_only=True)
class RandomFilterDatasetOutput(ModelInput, ModelOutput):
    pass


class RandomFilterDataset(Dataset):
    def __init__(self, config: RandomFilterDatasetConfig) -> None:
        self.config = config

    def __getitem__(
        self,
        item: int | slice,
    ) -> Optional[RandomFilterDatasetOutput]:
        match type(item):
            case builtins.int:
                if item >= len(self):
                    raise IndexError("dataset index out of range")

                batch_size = 1

            case builtins.slice:
                start = item.start if item.start is not None else 0
                stop = item.stop if item.stop is not None else len(self)
                step = item.step if item.step is not None else 1

                if abs(start) >= len(self):
                    raise IndexError("dataset index out of range")

                if abs(stop) > len(self):
                    raise IndexError("dataset index out of range")

                if step <= 0:
                    raise IndexError("dataset step is not valid")

                start %= len(self)
                stop %= len(self) + 1

                batch_size = round(abs(stop - start) / step)

                if not batch_size:
                    return None

            case _:
                raise TypeError("index must be int or slice")

        config = self.config

        samples = config.sections * batch_size

        p = config.pdf_p(samples)

        if config.all_pass:
            r = 1 / p.abs()
            theta = p.angle()

            z = r * torch.exp(1j * theta)

        else:
            z = config.pdf_z(samples)

        if not isinstance(item, int):
            z = rearrange(z, "(batch z) -> batch z", batch=batch_size)
            p = rearrange(p, "(batch p) -> batch p", batch=batch_size)

        z = order_sections(z, down_order=config.down_order, dim=-1)
        p = order_sections(p, down_order=config.down_order, dim=-1)

        z = construct_sections(z, config.sections, conjugate_pairs=True)
        p = construct_sections(p, config.sections, conjugate_pairs=True)

        k = (
            (torch.norm(p, dim=-1) / torch.norm(z, dim=-1)).squeeze()
            if config.all_pass
            else torch.ones(z.shape[:-1], dtype=z.real.dtype)
        )

        w, h = freqz_zpk(z, p, k, N=config.dft_bins, whole=config.whole_dft)

        return RandomFilterDatasetOutput(h=h, w=w, z=z, p=p, k=k)

    def __len__(self) -> int:
        config = self.config

        return config.batch_count * config.batch_size
