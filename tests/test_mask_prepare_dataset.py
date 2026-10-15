import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.mask_models.prepare_dataset import prepare_record
from src.mask_models.preprocessing import SegFormerMaskConfig


class MaskPreparationTests(unittest.TestCase):
    def test_resume_validates_complete_categorical_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); videos=root/"videos"; output=root/"output"; videos.mkdir(); (videos/"01.mp4").touch()
            target=output/"01"/"shot_01_shot_0000_controls_encoded.npz"; target.parent.mkdir(parents=True)
            mask=np.zeros((1,1,2,8,8),dtype=np.uint8); mask[:,:,1]=12
            np.savez_compressed(target,mask_encoded=mask)
            record={"video_id":"01","shot_id":"01_shot_0000","segment_start_frame":0,"segment_end_frame":8}
            with patch("src.mask_models.prepare_dataset.read_selected_rgb",return_value=(np.zeros((2,8,8,3),dtype=np.uint8),[0,7])):
                result=prepare_record(record,videos,output,SegFormerMaskConfig(num_frames=2,output_size=(8,8)),None,None,device="cpu",resume=True)
            self.assertTrue(result["resumed"]); self.assertEqual(result["dtype"],"uint8"); self.assertEqual(result["class_count"],2)

    def test_resume_rejects_wrong_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); videos=root/"videos"; output=root/"output"; videos.mkdir(); (videos/"01.mp4").touch()
            target=output/"01"/"shot_01_shot_0000_controls_encoded.npz"; target.parent.mkdir(parents=True); np.savez_compressed(target,wrong=np.zeros(1))
            record={"video_id":"01","shot_id":"01_shot_0000","segment_start_frame":0,"segment_end_frame":8}
            with patch("src.mask_models.prepare_dataset.read_selected_rgb",return_value=(np.zeros((2,8,8,3),dtype=np.uint8),[0,7])):
                with self.assertRaises(ValueError): prepare_record(record,videos,output,SegFormerMaskConfig(num_frames=2,output_size=(8,8)),None,None,device="cpu",resume=True)

if __name__ == "__main__": unittest.main()
