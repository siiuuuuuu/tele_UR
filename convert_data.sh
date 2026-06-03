# bash convert_data.sh


save_img=1
save_depth=0
save_wrist_img=1

demo_path=/home/lrz/dp_data/task1_expertdata
save_path=/home/lrz/dp_data/zarr_task1


python convert_demos.py --demo_dir ${demo_path} \
                                --save_dir ${save_path} \
                                --save_img ${save_img} \
                                --save_depth ${save_depth} \
                                --save_wrist_img ${save_wrist_img}
