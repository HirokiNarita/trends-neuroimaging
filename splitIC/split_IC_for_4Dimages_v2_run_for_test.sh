for num in {0..54}; do

    /home/hiroki/anaconda3/bin/python3 /media/hiroki/working/kaggle/trends-neuroimaging/splitIC/split_IC_for_4Dimages_v2_for_test.py $num
    $num = $((num))
    echo $num":complete"

done