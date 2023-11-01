import supervisely as sly
import os
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.fs import get_file_name, file_exists, dir_exists
import xml.etree.ElementTree as ET

from tqdm import tqdm

def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    train_path = os.path.join("archive","training_data","training_data")
    test_path = os.path.join("archive","test_data_images","test_data_images")
    batch_size = 30
    ds_name = "ds"
    images_folder = "images"
    bboxes_folder = "labels"
    images_ext = ".jpg"
    ann_ext = ".xml"

    ds_name_to_data = {"train": train_path, "test": test_path}


    def create_ann(image_path):
        labels = []

        file_name = get_file_name(image_path)

        ann_path = os.path.join(bboxes_path, file_name + ann_ext)
        if file_exists(ann_path):
            tree = ET.parse(ann_path)
            root = tree.getroot()

            img_height = int(root.find(".//height").text)
            img_wight = int(root.find(".//width").text)

            ann_objects = root.findall(".//object")
            for curr_object in ann_objects:
                obj_class_name = curr_object[0].text
                obj_class = name_to_class.get(obj_class_name)
                left = float(curr_object[1][0].text)
                top = float(curr_object[1][1].text)
                right = float(curr_object[1][2].text)
                bottom = float(curr_object[1][3].text)

                rect = sly.Rectangle(left=left, top=top, right=right, bottom=bottom)
                label = sly.Label(rect, obj_class)
                labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)


    car = sly.ObjClass("car", sly.Rectangle)
    pool = sly.ObjClass("pool", sly.Rectangle)
    name_to_class = {"1": car, "2": pool}
    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[car, pool])
    api.project.update_meta(project.id, meta.to_json())

    for ds_name, ds_path in ds_name_to_data.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        images_path = os.path.join(ds_path, images_folder)
        bboxes_path = os.path.join(ds_path, bboxes_folder)

        images_names = os.listdir(images_path)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(images_path, image_name) for image_name in images_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            if dir_exists(bboxes_path):
                anns = [create_ann(image_path) for image_path in img_pathes_batch]
                api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    return project
