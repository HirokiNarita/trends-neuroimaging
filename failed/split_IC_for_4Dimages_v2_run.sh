for num in {0..54}; do

    /home/hiroki/anaconda3/bin/python3 /media/hiroki/working/kaggle/trends-neuroimaging/split_IC_for_4Dimages_v2.py $num
    $num = $((num+1))
    echo $num":complete"

done
