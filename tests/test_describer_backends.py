from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PIL import Image

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.vlm_describer import describer
from monitoreo_urbano.vlm_describer.describer import (
    LLaVADescriber,
    Qwen2VLDescriber,
    VLMDescriber,
    _parse_response,
    _trim_unfinished,
    collapse_repetitions,
    MAX_FRAMES,
    MAX_NEW_TOKENS,
    _sample_frames,
    _to_pil_rgb,
    build_describer,
)


def _frames(count: int) -> list[np.ndarray]:
    return [np.full((6, 8, 3), index, dtype=np.uint8) for index in range(count)]


def test_parse_response_reads_description_and_attributes():
    raw = 'DESCRIPCION: Un auto rojo cruza la esquina.\nATRIBUTOS: {"vehicle_colors": ["rojo"], "direction": "norte"}'

    result = _parse_response(raw)

    assert result.description_nl == "Un auto rojo cruza la esquina."
    assert result.attributes.vehicle_colors == ["rojo"]
    assert result.attributes.direction == "norte"


def test_parse_response_without_the_expected_format_keeps_the_free_text():
    result = _parse_response("  Se ve una calle con poco movimiento.  ")

    assert result.description_nl == "Se ve una calle con poco movimiento."
    assert result.attributes.vehicle_colors == []


def test_parse_response_ignores_attributes_that_are_not_valid_json():
    result = _parse_response("DESCRIPCION: Hay humo.\nATRIBUTOS: {no es json}")

    assert result.description_nl == "Hay humo."
    assert result.attributes.behavior_tags == []


def test_parse_response_joins_a_direction_given_as_a_list():
    result = _parse_response('DESCRIPCION: Dos autos.\nATRIBUTOS: {"direction": ["norte", "este"]}')

    assert result.attributes.direction == "norte, este"


def test_parse_response_discards_attributes_that_do_not_match_the_schema():
    result = _parse_response('DESCRIPCION: Un auto.\nATRIBUTOS: {"vehicle_colors": "no es una lista"}')

    assert result.description_nl == "Un auto."
    assert result.attributes.vehicle_colors == []


def test_sample_frames_keeps_short_clips_untouched():
    frames = _frames(4)

    assert _sample_frames(frames, max_frames=6) is frames


def test_sample_frames_spreads_the_selection_over_a_long_clip():
    sampled = _sample_frames(_frames(12), max_frames=6)

    assert [int(frame[0, 0, 0]) for frame in sampled] == [0, 2, 4, 6, 8, 10]


def test_frames_are_converted_from_bgr_to_rgb_for_the_model():
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255

    image = _to_pil_rgb(bgr)

    assert isinstance(image, Image.Image)
    assert image.getpixel((0, 0)) == (0, 0, 255)


def test_build_describer_rejects_an_unknown_backend(monkeypatch):
    monkeypatch.setattr(settings, "vlm_backend", "inexistente")

    with pytest.raises(ValueError, match="inexistente"):
        build_describer()


def test_llava_backend_is_declared_but_not_implemented(monkeypatch):
    monkeypatch.setattr(settings, "vlm_backend", "llava")

    with pytest.raises(NotImplementedError, match="LLaVA"):
        build_describer()


def test_llava_describe_is_not_implemented():
    describer_instance = LLaVADescriber.__new__(LLaVADescriber)

    with pytest.raises(NotImplementedError):
        describer_instance.describe(_frames(1), "Cámara", ["car"])


def test_abstract_describer_defines_the_interface():
    class Incomplete(VLMDescriber):
        def describe(self, frames, camera_label, object_classes, trigger_reason=None, avoid_terms=None):
            return super().describe(frames, camera_label, object_classes, trigger_reason, avoid_terms)

    with pytest.raises(NotImplementedError):
        Incomplete().describe(_frames(1), "Cámara", ["car"])


def test_build_describer_creates_the_qwen_backend(monkeypatch):
    monkeypatch.setattr(settings, "vlm_backend", "qwen2-vl")
    created = []
    monkeypatch.setattr(describer, "Qwen2VLDescriber", lambda: created.append(True) or "modelo")

    assert build_describer() == "modelo"
    assert created == [True]


class FakeInputs(dict):
    def to(self, _device):
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.prompt_messages = None
        self.images = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.prompt_messages = messages
        return "texto del prompt"

    def __call__(self, text, return_tensors, images=None, videos=None):
        self.images = images
        self.videos = videos
        return FakeInputs(input_ids=np.zeros((1, 5), dtype=np.int64))

    def batch_decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        self.decoded_shape = ids.shape
        return ['DESCRIPCION: Choque entre dos autos.\nATRIBUTOS: {"vehicle_colors": ["gris"]}']


class FakeModel:
    device = "cpu"

    def generate(self, input_ids, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        return np.zeros((1, 9), dtype=np.int64)


def _install_fake_libraries(monkeypatch, processor: FakeProcessor, model: FakeModel) -> dict:
    loaded = {}
    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    transformers = types.ModuleType("transformers")

    class AutoProcessor:
        @staticmethod
        def from_pretrained(name, **kwargs):
            loaded["processor"] = (name, kwargs)
            return processor

    class Qwen2VLForConditionalGeneration:
        @staticmethod
        def from_pretrained(name, **kwargs):
            loaded["model"] = (name, kwargs)
            return model

    transformers.AutoProcessor = AutoProcessor
    transformers.Qwen2VLForConditionalGeneration = Qwen2VLForConditionalGeneration
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return loaded


def test_qwen_loads_the_model_on_the_requested_device(monkeypatch):
    loaded = _install_fake_libraries(monkeypatch, FakeProcessor(), FakeModel())

    Qwen2VLDescriber(model_name="qwen-falso", device="cuda:1")

    assert loaded["processor"][0] == "qwen-falso"
    assert loaded["model"][1]["device_map"] == "cuda:1"
    assert loaded["model"][1]["torch_dtype"] == "float16"


def test_qwen_loads_in_four_bits_only_when_asked(monkeypatch):
    loaded = _install_fake_libraries(monkeypatch, FakeProcessor(), FakeModel())
    import sys as _sys

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    _sys.modules["transformers"].BitsAndBytesConfig = BitsAndBytesConfig

    Qwen2VLDescriber(model_name="qwen-falso", device="cuda:0")
    assert "quantization_config" not in loaded["model"][1]

    monkeypatch.setattr(settings, "vlm_quantization", "bnb4")
    Qwen2VLDescriber(model_name="qwen-falso", device="cuda:0")
    config = loaded["model"][1]["quantization_config"]
    assert config.kwargs["load_in_4bit"] is True and config.kwargs["bnb_4bit_quant_type"] == "nf4"


def test_qwen_generates_only_once_even_when_the_output_never_closes(monkeypatch):
    processor, model = FakeProcessor(), FakeModel()
    _install_fake_libraries(monkeypatch, processor, model)
    qwen = Qwen2VLDescriber(model_name="qwen-falso", device="cpu")
    calls = []

    def looping_generate(input_ids, max_new_tokens):
        calls.append(max_new_tokens)
        return np.zeros((1, 5 + MAX_NEW_TOKENS), dtype=np.int64)

    model.generate = looping_generate
    processor.batch_decode = lambda ids, **kwargs: ["DESCRIPCION: Dos autos chocaron. Dos autos chocaron. Quedó cortada la"]

    result = qwen.describe(_frames(12), "Av. Libertador", ["car"], trigger_reason="accident")

    assert calls == [MAX_NEW_TOKENS]
    assert result.description_nl == "Dos autos chocaron."


def test_repeated_sentences_are_collapsed_and_the_unfinished_tail_is_trimmed():
    looped = "Un auto chocó contra un camión. El camión quedó detenido. El camión quedó detenido. El camión quedó detenido. El auto"

    assert collapse_repetitions(looped) == "Un auto chocó contra un camión. El camión quedó detenido. El auto"
    assert _trim_unfinished(collapse_repetitions(looped)) == "Un auto chocó contra un camión. El camión quedó detenido."
    assert _trim_unfinished("Termina bien.") == "Termina bien."


def test_qwen_describes_a_clip_sampling_frames_and_trimming_the_prompt(monkeypatch):
    processor, model = FakeProcessor(), FakeModel()
    _install_fake_libraries(monkeypatch, processor, model)
    qwen = Qwen2VLDescriber(model_name="qwen-falso", device="cpu")

    result = qwen.describe(_frames(12), "Av. Libertador", ["car"], trigger_reason="accident")

    assert result.description_nl == "Choque entre dos autos."
    assert result.attributes.vehicle_colors == ["gris"]
    assert len(processor.images) == MAX_FRAMES
    assert model.max_new_tokens == MAX_NEW_TOKENS
    assert processor.decoded_shape == (1, 4)
    content = processor.prompt_messages[1]["content"]
    assert [part["type"] for part in content].count("image") == MAX_FRAMES
    assert "Av. Libertador" in content[-1]["text"]
