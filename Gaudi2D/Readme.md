参考指南：https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion

```
docker start sd

docker run -it --name sd --runtime=habana -v /home/sd/:/data/ -e "http_proxy=$http_proxy" -e "https_proxy=$https_proxy" -e "no_proxy=localhost,127.0.0.1" -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest

docker exec -it sd bash

cd /data/optimum-habana/examples/stable-diffusion

python text_to_image_generation.py --model_name_or_path /data/stable-diffusion-2-1-base --prompts "An image of a squirrel in Picasso style" --num_images_per_prompt 28 --batch_size 1 --height 512 --width 512 --image_save_dir /data/stable_diffusion_images --use_habana --use_hpu_graphs --gaudi_config /data/stable-diffusion --bf16 --num_inference_steps 20

python text_to_image_generation.py --model_name_or_path /data/stable-diffusion-2-1-base --prompts "An image of a squirrel in Picasso style" --num_images_per_prompt 112 --batch_size 4 --height 512 --width 512 --image_save_dir /data/stable_diffusion_images --use_habana --use_hpu_graphs --gaudi_config /data/stable-diffusion --bf16 --num_inference_steps 20

```

查看当前Gaudi2D状态
```
watch -n 1 hl-smi
```

如果支持虚拟化，会出现SROV的信息。
```
lspci |grep Habana
~# lspci -s 18:00.0 -vvv
```

