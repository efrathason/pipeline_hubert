import logging
import shutil
import tarfile
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

parts = ["train", "test"]

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
        root = verbit_root / split
        recordings = RecordingSet.from_recordings(
            Recording.from_file(p) for p in (root / "audio").glob("*.mp3")
        )
        stms = list((root / "stm").glob("*.stm"))
        assert len(stms) == len(recordings), (
            f"Mismatch: found {len(recordings)} "
            f"sphere files and {len(stms)} STM files. "
            f"You might be missing some parts of TEDLIUM..."
        )
        
        segments = []
        for p in stms:
            with p.open() as f:
                for idx, l in enumerate(f):
                    rec_id, side, id, start, end, *words = l.split()
                    start, end = float(start), float(end)
                    text = " ".join(words)
                    segments.append(
                        SupervisionSegment(
                            id=f"{rec_id}-{idx}",
                            recording_id=rec_id,
                            start=start,
                            duration=round(end - start, ndigits=8),
                            channel=0,
                            text=text,
                            language="Farsi",
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
    verbit_root= "/export/c07/Babylon/mini_verbit_farsi", output_dir = "/export/c07/Babylon/output_files/mini_verbit_farsi") 