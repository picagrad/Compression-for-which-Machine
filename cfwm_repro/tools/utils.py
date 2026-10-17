import argparse
import warnings


class MlFLowBoolWorkaround(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, bool):
            out = values
        elif values.lower() in ('yes', 'true', 't', 'y', '1'):
            out = True
        elif values.lower() in ('no', 'false', 'f', 'n', '0'):
            out = False
        else:   # argparse's usage error (exit 2), not a traceback
            parser.error(f'argument {option_string}: expected True or False, got {values!r}')
        setattr(namespace, self.dest, out)



def get_embed_dims(task_model: str, cut_point: int) -> int:
    '''
    Simple helper function for getting the embedding dimensions of a swin backbone by model name.
    Currently only supports swin-l and swin-t as those are the only ones used in training codecs.
    '''
    if 'swin-l' in task_model.lower():
        return 192
    return 96 
    # raise NotImplementedError("Future feature. At this stage, embed dims must be calculated from model architecture, or via exprimentation")

def get_codec_type(uses_swin: bool, codec_family: str, var_rate: str) -> type:
    """The codec class. ELFC with GQVRF is the only codec this repository ships (both fixed by the entry points'
    set_defaults): QVRFELFCForSWIN for a SWIN backbone (UnProject reshapes its tokens to a feature map), QVRFELFC
    for a ConvNet backbone."""
    # 'elfc' is the only family.
    if codec_family != 'elfc' or var_rate != 'gqvrf':
        raise ValueError(f'only codec_family=elfc with var_rate=gqvrf is supported, got {codec_family!r}, {var_rate!r}')
    from cfwm_repro.codecs import QVRFELFC, QVRFELFCForSWIN
    return QVRFELFCForSWIN if uses_swin else QVRFELFC

def make_safe_random_choice(transforms, args):
    all_crop = []
    out_transforms = []
    for t_list in transforms:
        cropped = False
        new_t_list = []
        for t in t_list:
            if 'Crop' in t['type']:
                cropped = True
                if 'crop_size' in t:
                    t['crop_size'] = (args.crop_h, args.crop_w)
                else:
                    raise ValueError(f'Got a Crop transform with the following keys {t.keys()}, missing "crop_size"')
                crop = t
            if 'Resize' in t['type']:
                if cropped:
                    continue
                if 'scale' in t and args.image_h is not None and args.image_w is not None:
                    if isinstance(t['scale'], tuple):
                        t['scale'] = (args.image_h, args.image_w)
            new_t_list.append(t)
        all_crop.append(cropped)
        out_transforms.append(new_t_list)
    if any(all_crop) and not all(all_crop):
        print('Some RandomChoice transforms have crops and some do not. This is not allowed. Adding crop to all options.')
        for t_list, cropped in zip(out_transforms, all_crop):
            if not cropped:
                t_list.append(crop)
    return out_transforms, any(all_crop)
            

def make_safe_pipeline(pipeline_list:list, args, is_seg:bool = False):
    """Bound the training images' size: one crop to (crop_h, crop_w), as the last size-changing step.

    This is the training pipeline every published codec was trained with (image_h = image_w = crop_h = crop_w = 512):
    - a resize given as ONE fixed size (a tuple ``scale``) is replaced by (image_h, image_w): e.g. the segmentors'
      RandomResize (2048, 512) and RetinaNet / Mask R-CNN / Cascade R-CNN Resize (1333, 800) become (512, 512);
      a resize given as a LIST of sizes (FasterRCNN, ATSS, every RandomChoiceResize) is left as it is, so those
      models train on full-size images and only the crop bounds them;
    - an existing crop gets crop_size (crop_h, crop_w); a model without one gets a RandomCrop appended before the
      packing step; inside a RandomChoice (the DETR family), every option crops once and a resize after the crop
      is dropped. No pipeline crops twice.
    So models do not all see their own configs' zoom level. Kept as is so the published codecs reproduce.
    """
    out_pipeline = []
    cropped = False
    for cfg_dict in pipeline_list:
        if 'Crop' in cfg_dict['type']:
            cropped = True            
            if 'crop_size' in cfg_dict and args.crop_h is not None and args.crop_w is not None:
                cfg_dict['crop_size'] = (args.crop_h, args.crop_w)
            else:
                raise ValueError(f'Got a Crop transform with the following keys {cfg_dict.keys()}, missing "crop_size"')
        elif 'Resize' in cfg_dict['type']:
            if cropped:
                continue # Not allowing any more resizing after crop to prevent memory issues
            if 'scale' in cfg_dict and args.image_h is not None and args.image_w is not None:
                if isinstance(cfg_dict['scale'], tuple):
                    cfg_dict['scale'] = (args.image_h, args.image_w)
        elif 'RandomChoice' in cfg_dict['type'] and args.crop_h is not None and args.crop_w is not None:
            print('RandomChoice found. Making sure all options have crop')
            if cropped:
                continue # Not allowing any more random choices after crop to prevent memory issues
            new_transforms, new_cropped = make_safe_random_choice(cfg_dict['transforms'], args)
            cfg_dict = {k:v if k != 'transforms' else new_transforms for k,v in cfg_dict.items()}
            cropped = cropped or new_cropped
        
        out_pipeline.append(cfg_dict)
    if not cropped and args.crop_h is not None and args.crop_w is not None:
        print('No crop found. Adding crop to pipeline')
        if is_seg:
            # mmseg's RandomCrop has no notion of bboxes -- it only takes crop_size
            # (+ optional cat_max_ratio/ignore_index, left at their transform defaults here),
            # unlike mmdet's RandomCrop which needs recompute_bbox/allow_negative_crop.
            out_pipeline.insert(-1,{'type': 'RandomCrop', 'crop_size': (args.crop_h, args.crop_w)})
        else:
            out_pipeline.insert(-1,{'type': 'RandomCrop', 'crop_size': (args.crop_h, args.crop_w),'crop_type':'absolute','recompute_bbox':True,'allow_negative_crop':True})
        # print(out_pipeline)
    return out_pipeline

def DelayedLRScheduler(initial_stage:int = 0, active_stage:int = 2):
    def decorator(sched):
        """
        Decorator to delay the start of a learning rate scheduler until a specified stage.
        """
        new_sched = type(f'Delayed{sched.__name__}', (sched,), {}) if not 'Delayed' in sched.__name__ else sched
        original_init = sched.__bases__[0].__init__ if sched.__name__ == new_sched.__name__ else sched.__init__

        def __init__(self, *args, **kwargs):
            """
            Initialize the delayed scheduler with the optimizer and Lightning module.
            """
            # These MUST be set before original_init: torch's _LRScheduler.__init__
            # calls _initial_step() -> self.step(), which dispatches to the patched
            # delayed_step below. Assigning them afterwards left delayed_step reading
            # attributes that did not exist yet, so every _LRScheduler subclass
            # (CosineAnnealingLR, StepLR, ...) raised AttributeError at construction --
            # i.e. --lr_scheduler cosine/step were both dead on arrival, and only
            # ReduceLROnPlateau worked, because it alone does not call _initial_step().
            self._active = False
            self.stage = initial_stage
            self.external_last_epoch = -1
            original_init(self, *args, **kwargs)
            # Re-sync to the wrapped scheduler's own notion of where it starts.
            self.external_last_epoch = self.last_epoch
        
        def delayed_step(self, *args, **kwargs):
            """
            Override the step method to delay the scheduler's activation.
            """
            # Check if we're in the trigger stage for the first time.
            # An explicit epoch= is deprecated upstream, but still has to be honoured
            # rather than left unassigned -- the previous version of this branch only
            # warned, so `epoch` stayed unbound and the next line raised
            # UnboundLocalError for any caller that did pass one.
            if kwargs.get('epoch') is None:
                epoch = self.external_last_epoch + 1
            else:
                warnings.warn('Passing an epoch directly is deprecated, see torch.optim.lr_scheduler for more details', UserWarning)
                epoch = kwargs['epoch']
            self.external_last_epoch = epoch

            if not self._active and self.stage >= active_stage:
                self._active = True
                self.epoch_offset = self.external_last_epoch
                self.last_epoch = 0  # Reset last_epoch to 0 for the wrapped scheduler
                print(f"Scheduler activated at external epoch {self.epoch_offset}. "
                      f"Wrapped scheduler will now count epochs from 0.")

            # If activated, we step the wrapped scheduler with a relative epoch count.
            if self._active:
                self.last_epoch = self.external_last_epoch - self.epoch_offset
                # NB: this was `if epoch in kwargs`, which tested the int epoch against
                # the dict's KEYS and was therefore always False -- the rewrite below
                # never happened, so an explicitly-passed epoch reached the wrapped
                # scheduler in absolute (not activation-relative) form.
                if kwargs.get('epoch') is not None:
                    kwargs['epoch'] = self.last_epoch
                self.orig_step(*args, **kwargs)
        new_sched.orig_step = sched.step
        new_sched.step = delayed_step
        new_sched.__init__ = __init__
        return new_sched
    return decorator
