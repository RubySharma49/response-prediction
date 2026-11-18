library(affy)
library(hgu219.db)
library(AnnotationDbi)

cel_files <- ReadAffy(celfile.path = "data/raw/EMTAB/E-MTAB-3218/")
expr <- rma(cel_files)

expr_matrix <- exprs(expr)

# Extracting probeid keys from annotation database
ks = keys(hgu219.db, keytype = "PROBEID")

# Map probeids to gene symbols
probe2gene <- select(
  hgu219.db,
  keys = ks,
  columns = c("SYMBOL", "GENENAME"),
  keytype = "PROBEID"
)

match_ids = match(rownames(expr_matrix), probe2gene$PROBEID)

probe_symbol = probe2gene[match_ids,]

identical(rownames(expr_matrix), probe_symbol$PROBEID)

microarry_expr = cbind(rownames(expr_matrix), probe_symbol[,c(2:3)], expr_matrix)
colnames(microarry_expr) = c("PROBEID", colnames(microarry_expr)[2:ncol(microarry_expr)])


write.csv(microarry_expr, file = "data/processed/EMTAB_expression_matrix.csv", row.names = FALSE)          
