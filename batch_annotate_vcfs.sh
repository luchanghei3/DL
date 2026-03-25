#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: $0 2B.vcf 3B.vcf ..."
  exit 1
fi

ANNOTATE_AWK="${ANNOTATE_AWK:-annotate_cancer.awk}"
CANCER_MAP_TSV="${CANCER_MAP_TSV:-cancer_map.tsv}"
COSMIC_GLOB="${COSMIC_GLOB:-../cosmic_by_chr/*}"

if [[ ! -f "$ANNOTATE_AWK" ]]; then
  echo "错误: 找不到 $ANNOTATE_AWK"
  exit 1
fi

if [[ ! -f "$CANCER_MAP_TSV" ]]; then
  echo "错误: 找不到 $CANCER_MAP_TSV"
  exit 1
fi

shopt -s nullglob
unique_site_files=( *_unique_sites.tsv )
if [[ ${#unique_site_files[@]} -eq 0 ]]; then
  echo "错误: 当前目录下没有 *_unique_sites.tsv"
  exit 1
fi

for vcf in "$@"; do
  if [[ ! -f "$vcf" ]]; then
    echo "跳过: $vcf 不存在"
    continue
  fi

  sample="$(basename "$vcf" .vcf)"
  outdir="${sample}_annotate_out"
  mkdir -p "$outdir"

  echo "[${sample}] 开始处理"

  # step 1: 提取 hgvs 列（col11）
  awk 'NR>1 {print $11}' "$vcf" > "$outdir/vcf.hgvs"

  # step 2: 生成 mutation -> 英文癌种
  : > "$outdir/variant_cancer_map.tsv"
  for f in "${unique_site_files[@]}"; do
    cancer="$(basename "$f" _unique_sites.tsv)"
    awk -F'\t' -v c="$cancer" '
      NR==FNR {a[$1]; next}
      ($23 in a) {print $23 "\t" c}
    ' "$outdir/vcf.hgvs" "$f"
  done > "$outdir/variant_cancer_map.tsv"

  # step 3: 英文癌种 -> 中文癌种
  awk -F'\t' 'NR==FNR {a[$1]=$2; next} {print $1 "\t" a[$2]}' \
    "$CANCER_MAP_TSV" "$outdir/variant_cancer_map.tsv" > "$outdir/variant_cancer_map_cn.tsv"

  # step 4: 生成 mutation -> Gene/Mutation type/AA
  : > "$outdir/variant_cancer_map.allcol.tsv"
  for f in "${unique_site_files[@]}"; do
    cancer="$(basename "$f" _unique_sites.tsv)"
    awk -F'\t' -v c="$cancer" '
      NR==FNR {a[$1]; next}
      ($23 in a) {print $0 "\t" c}
    ' "$outdir/vcf.hgvs" "$f"
  done > "$outdir/variant_cancer_map.allcol.tsv"

  awk -F'\t' '{
      key = $23"\t"$1"\t"$12
      if (!(key in seen)) {
          seen[key] = $11
      } else if (seen[key] !~ "(^|,)"$11"(,|$)") {
          seen[key] = seen[key] "," $11
      }
  }
  END {
      for (k in seen) print k "\t" seen[k]
  }' "$outdir/variant_cancer_map.allcol.tsv" | sort > "$outdir/variant_cancer_map.infocol.tsv"

  # step 5: 匹配 cosmic control
  awk 'NR==FNR {a[$1]; next} ($1 in a)' "$outdir/vcf.hgvs" $COSMIC_GLOB > "$outdir/matched_cosmiccontrol.tsv"

  # step 6: 根据 mutation 从当前 vcf 抽取行
  awk 'NR==FNR {a[$1]; next} ($11 in a)' "$outdir/variant_cancer_map_cn.tsv" "$vcf" > "$outdir/cancer.vcf"
  sort "$outdir/cancer.vcf" | uniq > "$outdir/sorted_cancer.vcf"

  # step 7: 抽取前 4 列需求源文件
  cut -f11,12,13,14 "$outdir/sorted_cancer.vcf" > "$outdir/sorted_cancer.af.new.vcf"

  # step 8: 注释生成最终结果
  awk -f "$ANNOTATE_AWK" \
    "$outdir/variant_cancer_map.infocol.tsv" \
    "$outdir/matched_cosmiccontrol.tsv" \
    "$outdir/variant_cancer_map_cn.tsv" \
    "$outdir/sorted_cancer.af.new.vcf" \
    > "$outdir/sorted_cancer.af.annotated.vcf"

  echo "[${sample}] 完成: $outdir/sorted_cancer.af.annotated.vcf"
done
