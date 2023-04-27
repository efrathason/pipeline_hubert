import logging
import shutil
import tarfile
import re
from pathlib import Path
from typing import Dict, Optional, Union

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.utils import Pathlike

parts = ["dev", "test", "train"]

def prepare_verbit(
    verbit_root: Pathlike, output_dir: Optional[Pathlike] = None
    ) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:

    """
    Prepare manifests for verbit corpus.
    :return: A dict with standard corpus splits containing the manifests.

    """
    verbit_root = Path(verbit_root)
    output_dir = Path(output_dir) if output_dir is not None else None
    corpus = {}
    for split in parts:
        print(split)
        root = verbit_root / split
        recordings = RecordingSet.from_recordings(
            Recording.from_file(p) for p in (root / "wav").glob("*.wav")
        )
        stms = list((root / "stm").glob("*.stm"))
        assert len(stms) == len(recordings), (
            f"Mismatch: found {len(recordings)} "
            f"wav files and {len(stms)} STM files. "
            f"You might be missing some parts of verbit..."
        )
        
        segments = []
        for p in stms:
            with p.open() as f:
                for idx, l in enumerate(f):
                    rec_id, side, id, start, end, saparator, *words = l.split()
                    start, end = float(start), float(end)
                    text = " ".join(words)
                    text = re.sub(r'\{.[A-Z ]*\}' ,'', text)
                    text = re.sub(r' +', ' ', text)
                    segments.append(
                        SupervisionSegment(
                            id=f"{rec_id}-{idx}",
                            recording_id=rec_id,
                            start=start,
                            duration=round(end - start, ndigits=8),
                            channel=0,
                            text=text,
                            language="Arabic",
                            speaker=rec_id,
                        )
                    )
        supervisions = SupervisionSet.from_segments(segments)
        recordings, supervisions = fix_manifests(recordings, supervisions)

        corpus[split] = {"recordings": recordings, "supervisions": supervisions}
        validate_recordings_and_supervisions(**corpus[split])

        if output_dir is not None:
            recordings.to_file(output_dir / f"verbit_recordings_{split}.jsonl.gz")
            supervisions.to_file(output_dir / f"verbit_supervisions_{split}.jsonl.gz")
    
    return corpus

prepare_verbit(
    verbit_root= "/outagescratch/skhudan1/eorenst1/verbit_arabic_splited", output_dir = "/outagescratch/skhudan1/eorenst1/supervision_verbit") 