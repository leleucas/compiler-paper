function mmdet_avg_ratio() {
  model_name=$1
  model_view_file=$2
  model_noview_file=$3
  model_nocoder_file=$4
  cat $model_view_file | grep "__benchmark_avg_iter_time" | cut -d ":" -f 4 > ${model_name}_tmp1.txt
  cat $model_noview_file | grep "__benchmark_avg_iter_time" | cut -d ":" -f 4 > ${model_name}_tmp2.txt
  cat $model_nocoder_file | grep "__benchmark_avg_iter_time" | cut -d ":" -f 4 > ${model_name}_tmp3.txt
  paste ${model_name}_tmp1.txt ${model_name}_tmp2.txt ${model_name}_tmp3.txt > ${model_name}_res.txt
  awk 'BEGIN{avg1=0;avg2=0;avg3=0;}{avg1=avg1+$1;avg2=avg2+$2;avg3=avg3+$3;print ($2-$1)/$2,($3-$1)/$3 >> "avg_tmp.txt"}END{print avg1/12,avg2/12,avg3/12,(avg2-avg1)/avg2,(avg3-avg1)/avg3 >> "avg_tmp.txt"}' ${model_name}_res.txt
  paste ${model_name}_res.txt avg_tmp.txt > tmp.txt
  mv tmp.txt ${model_name}_res.txt
  sed -e "1i time_of_8threadswithview time_of_8threadwithoutview time_of_8threadwithoutcoder ratio_of_perf_promotionofview ratio_of_perf_promotionofcoder" ${model_name}_res.txt > tmp.txt
  sed -e "\$i avg_time_of_8threadswithview avg_time_of_8threadwithoutview avg_time_of_8threadwithoutcoder avg_ratio_of_perf_promotionofview avg_ratio_of_promotionofcoder" tmp.txt > ${model_name}_res.txt
  rm avg_tmp.txt tmp.txt ${model_name}_tmp1.txt ${model_name}_tmp2.txt ${model_name}_tmp3.txt
  # echo $model_name $model_view_file $model_noview_file
}

function detr_avg_ratio() {
  model_name=$1
  model_view_file=$2
  model_noview_file=$3
  model_nocoder_file=$4
  cat $model_view_file | grep "__benchmark_avg_iter_time" | cut -d ' ' -f 2 > ${model_name}_tmp1.txt
  cat $model_noview_file | grep "__benchmark_avg_iter_time" | cut -d ' ' -f 2 > ${model_name}_tmp2.txt
  cat $model_nocoder_file | grep "__benchmark_avg_iter_time" | cut -d ' ' -f 2 > ${model_name}_tmp3.txt
  paste ${model_name}_tmp1.txt ${model_name}_tmp2.txt ${model_name}_tmp3.txt > ${model_name}_res.txt
  awk 'BEGIN{avg1=0;avg2=0;avg3=0;}{avg1=avg1+$1;avg2=avg2+$2;avg3=avg3+$3;print ($2-$1)/$2, ($3-$1)/$3 >> "avg_tmp.txt"}END{print avg1/9,avg2/9,avg3/9,(avg2-avg1)/avg2,(avg3-avg1)/avg3 >> "avg_tmp.txt"}' ${model_name}_res.txt
  paste ${model_name}_res.txt avg_tmp.txt > tmp.txt
  mv tmp.txt ${model_name}_res.txt
  sed -e "1i time_of_8threadswithview time_of_8threadwithoutview time_of_8threadswithoutcoder ratio_of_perf_promotionofview ratio_of_perf_promotionofcoder" ${model_name}_res.txt > tmp.txt
  sed -e "\$i avg_time_of_8threadswithview avg_time_of_8threadwithoutview avg_time_of_8threadwithoutcoder avg_ratio_of_perf_promotionofview avg_ratio_of_perf_promotionofcoder" tmp.txt > ${model_name}_res.txt
  rm avg_tmp.txt tmp.txt ${model_name}_tmp1.txt ${model_name}_tmp2.txt ${model_name}_tmp3.txt
  # echo $model_name $model_view_file $model_noview_file
}

function res_avg_ratio() {
  model_name=$1
  awk 'BEGIN{avg1=0;avg2=0;avg3=0;}{avg1=avg1+$1;avg2=avg2+$2;avg3=avg3+$3;print ($2-$1)/$2,($3-$1)/$3 >> "avg_tmp.txt"}END{print avg1/10,avg2/10,avg3/10,(avg2-avg1)/avg2,(avg3-avg1)/avg3 >> "avg_tmp.txt"}' ${model_name}_res.txt
  paste ${model_name}_res.txt avg_tmp.txt > tmp.txt
  mv tmp.txt ${model_name}_res.txt
  sed -e "1i time_of_8threadswithview time_of_8threadwithoutview time_of_8threadwithoutcoder ratio_of_perf_promotionofview ratio_of_perf_promotionofcoder" ${model_name}_res.txt > tmp.txt
  sed -e "\$i avg_time_of_8threadswithview avg_time_of_8threadwithoutview avg_time_of_8threadwithoutcoder avg_ratio_of_perf_promotionofview avg_ratio_of_promotionofcoder" tmp.txt > ${model_name}_res.txt
  rm avg_tmp.txt tmp.txt
  # echo $model_name $model_view_file $model_noview_file
}


if [ $1 == "mask_rcnn" -o $1 == "ssd" ]; then
  mmdet_avg_ratio $1 $2 $3 $4
elif [ $1 == "detr" ]; then
  detr_avg_ratio $1 $2 $3 $4
elif [ $1 == "mmdet" ]; then
  res_avg_ratio mask_rcnn
  res_avg_ratio ssd
  res_avg_ratio detr
fi
