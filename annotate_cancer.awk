BEGIN {
    FS = OFS = "\t"
}

function emit_row(    key,g,mt,aa,ac2,an2,af2,cancer_type,genome_ad) {
    key = $1
    g = (key in gene ? gene[key] : "")
    mt = (key in muttype ? muttype[key] : "")
    aa = (key in aachg ? aachg[key] : "")
    ac2 = (key in gad_ac ? gad_ac[key] : "")
    an2 = (key in gad_an ? gad_an[key] : "")
    af2 = (key in gad_af ? gad_af[key] : "")

    cancer_type = (NF >= 5 ? $5 : "")
    if (cancer_type == "" && key in cancer_cn) cancer_type = cancer_cn[key]

    genome_ad = (NF >= 6 ? $6 : "")

    print $1, $2, $3, $4, g, mt, aa, cancer_type, genome_ad, ac2, an2, af2
}

# File 1: variant_cancer_map.infocol.tsv
ARGIND == 1 {
    if (FNR == 1 && $1 == "Mutation") next
    key = $1
    gene[key] = $2
    muttype[key] = $3
    aachg[key] = $4
    next
}

# File 2: matched_cosmiccontrol.tsv
ARGIND == 2 {
    if (FNR == 1 && $1 == "Mutation") next
    key = $1
    gad_ac[key] = $3
    gad_an[key] = $4
    gad_af[key] = $5
    next
}

# File 3 can be either:
#   A) sorted_cancer.af.new.vcf (3-file mode)
#   B) variant_cancer_map_cn.tsv (4-file mode)
ARGIND == 3 {
    if (FNR == 1) {
        third_is_sorted = (NF >= 4)
        if (third_is_sorted) {
            print "Mutation", "Minor", "Frequence", "Total", "Gene", "Mutation type", "Amino Acid Change", "Cancer type", "Genome AD", "GAD_AC", "GAD_AN", "GAD_AF"
            if ($1 == "Mutation" && $2 == "Minor") next
            emit_row()
            next
        }
    }

    if (third_is_sorted) {
        emit_row()
    } else {
        key = $1
        if (!(key in cancer_cn) || cancer_cn[key] == "") cancer_cn[key] = $2
    }
    next
}

# File 4: sorted_cancer.af.new.vcf (when file 3 is variant_cancer_map_cn.tsv)
ARGIND == 4 {
    if (FNR == 1) {
        print "Mutation", "Minor", "Frequence", "Total", "Gene", "Mutation type", "Amino Acid Change", "Cancer type", "Genome AD", "GAD_AC", "GAD_AN", "GAD_AF"
        if ($1 == "Mutation" && $2 == "Minor") next
    }

    emit_row()
    next
}
