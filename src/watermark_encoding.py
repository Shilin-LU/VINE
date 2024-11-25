import os, torch, time, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from vine_turbo import VINE_Turbo
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms


def crop_to_square(image):
    width, height = image.size

    min_side = min(width, height)
    left = (width - min_side) // 2
    top = (height - min_side) // 2
    right = left + min_side
    bottom = top + min_side

    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image
    
def main(args, device):
    ### ============= load model =============
    watermark_encoder = VINE_Turbo.from_pretrained(args.pretrained_model_name)
    watermark_encoder.to(device)

    ### ============= load image =============  
    input_image_pil = Image.open(args.input_path).convert('RGB') # 512x512 
    if input_image_pil.size[0] != input_image_pil.size[1]:
        input_image_pil = crop_to_square(input_image_pil)
    
    size = input_image_pil.size
    t_val_256 = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), 
        transforms.ToTensor(),
    ])
    t_val_512 = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    ])
    resized_img = t_val_256(input_image_pil) # 256x256
    resized_img = 2.0 * resized_img - 1.0
    input_image = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device) # 512x512
    input_image = 2.0 * input_image - 1.0
    resized_img = resized_img.unsqueeze(0).to(device)

    ### ============= load message =============
    if args.load_text: # text to bits
        if len(args.message) > 12:
            print('Error: Can only encode 100 bits (12 characters)')
            raise SystemExit
        data = bytearray(args.message + ' ' * (12 - len(args.message)), 'utf-8')
        packet_binary = ''.join(format(x, '08b') for x in data)
        watermark = [int(x) for x in packet_binary]
        watermark.extend([0, 0, 0, 0])
        watermark = torch.tensor(watermark, dtype=torch.float).unsqueeze(0)
        watermark = watermark.to(device)
    else: # random bits
        data = torch.randint(0, 2, (100,))
        watermark = torch.tensor(data, dtype=torch.float).unsqueeze(0)
        watermark = watermark.to(device)

    ### ============= watermark encoding =============
    start_time = time.time()
    encoded_image_256 = watermark_encoder(resized_img, watermark)
    end_time = time.time()
    print('\nEncoding time:', end_time - start_time, 's', '\n (Note that please execute multiple times to get the average time)\n')

    ### ============= resolution scaling to original size =============
    residual_256 = encoded_image_256 - resized_img # 256x256
    residual_512 = t_val_512(residual_256) # 512x512 or original size
    encoded_image = residual_512 + input_image # 512x512 or original size
    encoded_image = encoded_image * 0.5 + 0.5
    encoded_image = torch.clamp(encoded_image, min=0.0, max=1.0)

    ### ============= save the output image =============
    output_pil = transforms.ToPILImage()(encoded_image[0])
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    save_loc = os.path.join(args.output_dir, os.path.split(args.input_path)[-1][:-4]+'_wm.png')
    output_pil.save(save_loc)
    print(f'\nWatermarked image saved at: {save_loc}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./example/input/2.png', help='path to the input image')
    parser.add_argument('--output_dir', type=str, default='./example/watermarked_img', help='the directory to save the output')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/VINE-R', help='path to the checkpoint')
    parser.add_argument('--pretrained_model_name', type=str, default='Shilin-LU/VINE-R-Enc', help='pretrained_model_name')
    parser.add_argument('--message', type=str, default='Hello World!', help='the secret message to be encoded')
    parser.add_argument('--load_text', type=bool, default=True, help='the flag to decide to use inputed text or random bits')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    main(args, device)
    
