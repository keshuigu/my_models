from .vqgan import VQModel, VQLPIPSWithDiscriminatorConfig, VQLPIPSWithDiscriminatorModel


def get_vqgan_model(config):
    assert config.model_name == 'vqgan'
    return VQModel(
        in_channels=config.model_in_channels,
        out_channels=config.model_out_channels,
        down_block_types=config.model_down_block_types,
        up_block_types=config.model_up_block_types,
        block_out_channels=config.model_block_out_channels,
        layers_per_block=config.model_layers_per_block,
        act_fn=config.model_act_fn,
        latent_channels=config.model_latent_channels,
        norm_num_groups=config.model_norm_num_groups,
        scaling_factor=config.model_scale_factor,
        num_vq_embeddings=config.model_num_vq_embeddings,
        vq_embed_dim=config.model_vq_embed_dim,
    )


def get_discriminator(config):
    assert config.model_discriminator_name == 'VQLPIPSWithDiscriminatorModel'
    disc_config = VQLPIPSWithDiscriminatorConfig(
        disc_conditional=config.model_disc_conditional,
        disc_in_channels=config.model_in_channels,
        disc_start=config.model_disc_start,
        disc_weight=config.model_disc_weight,
        codebook_weight=config.model_codebook_weight,
        disc_num_layers=config.model_disc_num_layers,
    )
    return VQLPIPSWithDiscriminatorModel(disc_config)
