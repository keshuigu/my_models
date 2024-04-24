from dataclasses import dataclass


@dataclass
class Vqgan:
    root_path = '/mnt/share/workspace'
    output_dir = root_path + '/research/my_models/outputs/vqgan'
    data_name = 'celeba'
    data_path = root_path + '/generate_models/dataset/celeba_prepared'
    data_filename = 'mini1000_1.txt'

    data_split_ratio = 0.8  # 训练和测试比例
    # 最终输入模型的大小和channel
    data_size = 128
    data_channels = 3

    # 裁剪图片
    data_crop_size = 108
    data_crop_offset = 0
    data_h_flip_p = 0.0
    data_crop = True

    # 训练配置
    train_loader_batchSize = 14
    train_loader_numWorkers = 4
    train_optim_lr = 1e-4
    train_optim_warmupSteps = 500
    train_epochsNum = 100

    # 测试配置
    test_loader_batchSize = 14
    test_loader_numWorkers = 4

    # 这是accelerate的库混合精度训练，还有梯度累积
    train_accelerate_mixedPrecision = "no"
    train_accelerate_gradientAccumulationSteps = 1
    # 评估
    train_evalPerEpoch = 1
    train_savePerEpochs = 1
    eval_batchSize = 2
    # 模型定义
    model_name = 'vqgan'
    model_vae_pretrained = None
    model_dis_pretrained = None
    model_in_channels = data_channels
    model_out_channels = data_channels
    # 下采样上采样
    model_down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D",)
    model_up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D",)
    # 每一层，长宽减半，最后一层不变，相当于这个长宽都/8
    model_block_out_channels = (128, 256, 256, 512)
    # 每个块中的层的数量
    model_layers_per_block = 2
    model_act_fn = "silu"  # Sigmoid Linear Unit
    # 隐含空间的通道数
    model_latent_channels = 4
    # TODO 模型中使用的归一化层的组数
    model_norm_num_groups = 16
    model_scale_factor = 0.18215
    # 嵌入空间向量数量
    model_num_vq_embeddings = 512
    # 维度 model_latent_channels = model_vq_embed_dim
    model_vq_embed_dim = 4

    model_discriminator_name = "VQLPIPSWithDiscriminatorModel"
    model_disc_conditional = False
    # Taming的GAN优化策略，调整生成器判别器训练速度用的
    model_disc_start = 60000
    model_disc_weight = 0.6
    model_codebook_weight = 1.0
    model_disc_num_layers = 4

    # TODO 每个Epoch跑5次，跑5个step就停
    debug = False
    debug_time = 5
