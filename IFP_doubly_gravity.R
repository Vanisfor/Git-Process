# 输入表3、4和5的数据
trip_productions <- c(A = 6253.729, B = 7638.051, C = 10374.335, D = 13226.158, E = 16998.700)
trip_attractions <- c(A = 4904.187, B = 10898.195, C = 16834.7292, D = 5994.007, E = 16347.292)
cost_matrix <- matrix(c(1, 5, 3, 4, 5,
                        5, 1, 5, 1, 3,
                        3, 5, 2, 3, 2,
                        4, 1, 3, 1, 4,
                        5, 3, 2, 4, 2), 
                      nrow = 5, ncol = 5, byrow = TRUE,
                      dimnames = list(c("A", "B", "C", "D", "E"), c("A", "B", "C", "D", "E")))

# 应用阻碍函数 f(cij) = cij^-0.7
deterrence_matrix <- cost_matrix ^ -0.7

# 初始化 OD 矩阵
initial_OD <- outer(trip_productions, trip_attractions, "*") * deterrence_matrix

# 定义使用矩阵范数判断法的 IPF 算法函数
ipf_algorithm_matrix_norm <- function(OD_matrix, productions, attractions, tol = 0.01, max_iter = 100) {
  iteration <- 0
  prev_OD <- OD_matrix  # 存储前一轮的 OD 矩阵
  converged <- FALSE
  
  while (!converged && iteration < max_iter) {
    # 行方向调整
    row_totals <- rowSums(OD_matrix)
    row_factors <- productions / row_totals
    OD_matrix <- sweep(OD_matrix, 1, row_factors, "*")
    
    # 列方向调整
    col_totals <- colSums(OD_matrix)
    col_factors <- attractions / col_totals
    OD_matrix <- sweep(OD_matrix, 2, col_factors, "*")
    
    # 计算矩阵范数差异（Frobenius 范数）
    norm_diff <- sqrt(sum((OD_matrix - prev_OD)^2))
    converged <- norm_diff < tol
    
    # 更新前一轮的 OD 矩阵
    prev_OD <- OD_matrix
    iteration <- iteration + 1
  }
  
  if (converged) {
    cat("Converged in", iteration, "iterations.\n")
  } else {
    cat("Did not converge within max iterations.\n")
  }
  
  return(round(OD_matrix, 1))  # 返回保留到小数点后1位的结果
}

# 应用 IPF 算法
final_OD_matrix <- ipf_algorithm_matrix_norm(initial_OD, trip_productions, trip_attractions)

# 输出结果
print("Final OD Matrix (Rounded to 1 decimal places):")
print(final_OD_matrix)


