
import sys
 
# setting path
sys.path.append('/home/eorenst1/pipeline_hubert')

import get_data_function as gdf

data_dir = "/home/eorenst1/pipeline_hubert/data"

manifest_dir = data_dir + "/manifest"
cuts_dir = data_dir + "/cuts"


print("create the manifest")
gdf.save_manifest(manifest_dir, download=False)

print("create the cuts from manifest")
gdf.save_data(manifest_dir, cuts_dir)
