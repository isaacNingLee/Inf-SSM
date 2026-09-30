
def freeze_model(args, model):

    # Unfreeze all layers
    for param in model.parameters():
            param.requires_grad = True

    if args.freeze_option == 'all':
        for param in model.parameters():
            param.requires_grad = False

    elif args.freeze_option == 'all_but_head':
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False

    elif args.freeze_option == 'blocks':

        
        if args.model.startswith('deit'):
            num_blocks = len(model.blocks)
        elif args.model.startswith('vim'):
            num_blocks = len(model.layers)

        # Calculate the number of blocks to freeze
        num_freeze = int(num_blocks * args.freeze_ratio)

        print(f'Freezing {num_freeze} out of {num_blocks} blocks')


        # Freeze the first half of the blocks

        if args.model.startswith('deit'):
            for idx in range(num_freeze):
                for param in model.blocks[idx].parameters():
                    param.requires_grad = False

        elif args.model.startswith('vim'):
            for idx in range(num_freeze):
                for param in model.layers[idx].parameters():
                    param.requires_grad = False


        # freeze patch embedding parameters
        for param in model.patch_embed.parameters():
            param.requires_grad = False
       

    elif args.freeze_option == 'patch_embed':


        # freeze patch embedding parameters
        for param in model.patch_embed.parameters():
            param.requires_grad = False

    elif args.freeze_option == 'None':
        pass
            