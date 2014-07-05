#!/bin/sh

main() {
  local csvfile=$1

  cat $csvfile | textql -dlm=";" -header=true -sql="SELECT img_width,beta,AVG(cpu_enc_perm),AVG(cpu_enc_chen),AVG(cpu_dec_chen),AVG(cpu_dec_perm),AVG(gpu_enc_perm),AVG(gpu_dec_chen),AVG(gpu_dec_perm),no_imgs_gpu_enc_chen,AVG(gpu_enc_chen) FROM tbl GROUP BY img_width,beta,no_imgs_gpu_enc_chen ORDER BY img_width ASC"
}

if [ "$#" != "1" ]; then
  echo "Usage:" "$0" "CSV-file"
  exit 1
else
  main
fi

