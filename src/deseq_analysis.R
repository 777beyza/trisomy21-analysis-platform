# src/deseq_analysis.R
# Bu script Python tarafindan subprocess ile cagrilacak.
# Input: data/rna_input.csv (Icerik: Gene, Control_1..3, Patient_1..3 vb.)
# Output: data/rna_results.csv (Icerik: Gene, Log2FC, P_value, Neg_Log10_P)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Kullanim: Rscript deseq_analysis.R <input.csv> <output.csv>")
}

input_file <- args[1]
output_file <- args[2]

# Veriyi oku
cat("R Script: Veri okunuyor: ", input_file, "\n")
data <- read.csv(input_file, stringsAsFactors = FALSE)

# Beklenen format: 1. kolon 'Gene', diger kolonlar sayim degerleri
# Ozetle, yarisi kontrol, yarisi hasta grubu kabul edelim
num_cols <- ncol(data)
if(num_cols < 3) {
  stop("Yeterli sayida kolon yok. En az 1 gen ve 2 sample kolonu olmali.")
}

n_samples <- num_cols - 1
mid <- floor(n_samples / 2)
control_cols <- 2:(mid + 1)
patient_cols <- (mid + 2):num_cols

cat("R Script: ", length(control_cols), " kontrol ve ", length(patient_cols), " hasta ornegi bulundu.\n")

results <- data.frame(
  Gene = character(nrow(data)),
  Log2FC = numeric(nrow(data)),
  P_value = numeric(nrow(data)),
  `_Log10_P` = numeric(nrow(data)),
  stringsAsFactors = FALSE
)

# Base R kullanarak her gen icin Welch Two Sample t-test ve Fold Change (log2)
for (i in 1:nrow(data)) {
  gene_name <- data[i, 1]
  
  control_vals <- as.numeric(data[i, control_cols])
  patient_vals <- as.numeric(data[i, patient_cols])
  
  # Log2 transformasyonunu yaklasik olarak yapiyoruz (+1 offset ile 0 hatasini onlemek icin)
  mean_control <- mean(log2(control_vals + 1))
  mean_patient <- mean(log2(patient_vals + 1))
  
  # Log2 Fold Change = (Log2 Patient) - (Log2 Control)
  log2fc <- mean_patient - mean_control
  
  # P-value (Eger varyans yoksa t.test patlar, try-catch ile yonetiyoruz)
  pval <- 1.0
  tryCatch({
    if(var(control_vals) > 0 || var(patient_vals) > 0) {
      t_res <- t.test(patient_vals, control_vals)
      pval <- t_res$p.value
    }
  }, error = function(e) {
    pval <- 1.0
  })
  
  # Eger pval 0'a asiri yakinsa, NA olmamasi icin min sınır koyalim
  if(is.na(pval) || pval == 0) pval <- 1e-10
  
  neg_log10_p <- -log10(pval)
  
  results$Gene[i] <- gene_name
  results$Log2FC[i] <- log2fc
  results$P_value[i] <- pval
  results$`-Log10_P`[i] <- neg_log10_p
}

# Sonuclari CSV'ye yaz
write.csv(results, output_file, row.names = FALSE, quote = FALSE)
cat("R Script: Analiz tamamlandi ve kaydedildi: ", output_file, "\n")
