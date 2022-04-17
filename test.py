import unittest
import shutil
import os

from example import *

class TestGeneratorsExecution(unittest.TestCase):
    def test_generators_execution(self):
        self.number_of_img = 1
        self.number_of_step = 2
        self.outdir = 'temp'
        self.video_name = "video"
        
        generators = init_generators()

        generate_images(generators, self.number_of_img, self.number_of_step, self.outdir, tqdm_disable=True)
        self.assertEqual(len(os.listdir(self.outdir)), self.number_of_img * self.number_of_step)

        create_video(self.outdir, self.video_name, self.number_of_img, self.number_of_step)       
        self.assertTrue(os.path.exists(f'{self.outdir}/{self.video_name}.mp4'))

        shutil.rmtree(self.outdir)


if __name__ == '__main__':
    unittest.main()