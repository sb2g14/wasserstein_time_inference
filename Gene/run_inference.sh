for i in 0
do
for j in 0
do
for k in 0
do
for l in 0
do
for ac in 1000
do
for iter in 1
do
for proc in 1
do
	nice python3 Parameter_inference_grid_time_gene.py $i $j $k $l $ac $iter &
done
done
done
done
done
done
done
