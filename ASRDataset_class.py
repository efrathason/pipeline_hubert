from lhotse import CutSet
from lhotse.dataset.input_strategies import AudioSamples
from lhotse.workarounds import Hdf5MemoryIssueFix
from lhotse.dataset.speech_recognition import validate_for_asr
from lhotse.dataset.collation import TokenCollater
import torch

class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: TokenCollater):
        self.tokenizer = tokenizer
        self.input_method = AudioSamples()
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)
    def __getitem__(self, cuts: CutSet) -> dict:
        validate_for_asr(cuts)
        self.hdf5_fix.update()
        # We want the minibatch sorted by length (longest to shorted)
        cuts = cuts.sort_by_duration(ascending=False)

        inputs, _ = self.input_method(cuts)
        supervision_intervals = self.input_method.supervision_intervals(cuts)
        tokens, token_lens = self.tokenizer(cuts)
        return {
            'input': inputs,
            'target': tokens,
            'lens': supervision_intervals,
            'text': [s.text for cut in cuts for s in cut.supervisions]
        }

    
    def move_to(b, device):
        return {
            'input': b['input'].to(device),
            'target': b['target'].to(device),
            'lens': {
                'sequence_idx': b['lens']['sequence_idx'].to(device),
                'start_sample': b['lens']['start_sample'].to(device),
                'num_samples': b['lens']['num_samples'].to(device)
            },
            'text': b['text'],
        }