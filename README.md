NST_PROG_TESTING.ipynb --------> testing  SNT renderer (prog_renderer.npz is a npz extracted from test image, it is alredy available in repo, or we can feed any npz file with knowledge of no of strokes)

NPZ_OPTIMIZER.ipynb ----------> testing renderer with npz file and then optimize to get clear image

json2npz.py  -----------> converting JSON to .npz format
          steps to run this file:
                1. clone to this repo TATA_RESEARCH
                2. !python json2npz.py --json stroke_params.json --out  my_strokes.npz
                3. this npz output can be fed to nst_prog_testing.ipynb to see the output as image

                
Trans_gan_training.ipynb ---------> training SNT with TransGan Architecture 
          some instructions:
                1.check points will be saved in checkpoints_G folder
                2.15 GB GPU required
             
