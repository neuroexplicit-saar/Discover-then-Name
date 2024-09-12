import webdataset as wds
import torch


class CustomDataCollatorImg:
    def __init__(self) -> None:
        pass

    def __call__(self, batch):
        imgs = [i["image"] for i in batch]
        idxs = [i['__key__'] for i in batch]
        imgs = torch.stack(imgs)
        return imgs, idxs


class CC3MImg:
    def __init__(self):
        pass

    def get_wds_dataset(self, input_shards, transform, batch_size, collator=None):
        """
        return a dataset that returns an image, and text
        """

        pipeline = [
            wds.SimpleShardList(input_shards),
            # at this point we have an iterator over all the shards
        ]

        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(),
                # wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb"),
                wds.rename(image="jpg;png;jpeg"),
                wds.map_dict(image=transform),])

        pipeline.extend(
            [wds.batched(batch_size, partial=False, collation_fn=collator)])

        dataset = wds.DataPipeline(*pipeline)
        return dataset

    def get_dataloader(self, dataset, batch_size=None, shuffle=False, num_workers=1):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader
