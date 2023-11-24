from prophecy.utils.misc import TrainSplit, ValSplit, UnseenSplit

SPLITS = {
    'train': TrainSplit(),
    'val': ValSplit(),
    'unseen': UnseenSplit()
}
