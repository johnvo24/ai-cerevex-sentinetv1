from utils.gdrive import GDrive

gdrive = GDrive()
gdrive.download_file(
  file_name='checkpoint-22500.zip',
  folder_path='/models/cerevex-sentinetv1',
  destination_path='/model'
)