#! /bin/bash


# python ogbn.py --datasets ogbn-arxiv --alpha 1.0  --step 0
# python ogbn.py --datasets ogbn-products --alpha 1.0  --step 0


for i in $(seq 2 1  4);
do 
    python ogbn.py  --datasets   ogbn-arxiv  --num_parts ${i}   --alpha 0.95 
done


# for i in $(seq 0 1  4);
# do 
#     for j in $(seq 0 1  4)
#     do
#         python ogbn_no_bag.py --datasets   ogbn-arxiv  --num_parts 2   --step ${i}  --rep ${j}
#     done
# done

# python ogbn.py --datasets ogbn-products --alpha   1.0    --step 0
# python ogbn.py --datasets ogbn-products --alpha   1.0    --step 1
# python ogbn.py --datasets ogbn-products --alpha   1.0    --step 2
# python ogbn.py --datasets ogbn-products --alpha   1.0    --step 3


# python ogbn_no_bag.py  --datasets ogbn-products --alpha   1.0    --step 0
# python ogbn_no_bag.py  --datasets ogbn-products --alpha   1.0    --step 1
# python ogbn_no_bag.py  --datasets ogbn-products --alpha   1.0    --step 2


# python stacking.py --alpha 0.9     --dataset vk  --step 5 

# python stacking.py --alpha 1.0    --dataset house  --step 50  


python ogbn.py  --datasets   ogbn-arxiv    --alpha 0.95 

