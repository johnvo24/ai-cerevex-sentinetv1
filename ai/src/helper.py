from utils.gdrive import GDrive

gdrive = GDrive()
file_id = gdrive.get_file_id(
  file_name='checkpoint-22500.zip', 
  folder_path='/models/cerevex-sentinetv1'
)
gdrive.download_file(
  file_id=file_id,
  destination_path='model/checkpoint-22500.zip'
)