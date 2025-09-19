using JSON
using Printf

const N = 5
const SUBMATRIX_SIZE = N - 1

"""
Converts an integer mask to its binary string representation.
"""
function mask_to_binary_string(mask::Int, len::Int)
    return string(mask, base=2, pad=len)
end

"""
Checks if a binary matrix A of a given size can be LU-factorized without pivoting
using Bareiss' algorithm, a fraction-free variant of Gaussian elimination.
"""
function can_lu_factorize(A::Matrix{Int64}, size::Int)
    if size <= 1
        return A[1, 1] != 0
    end
    if size > N
        return false
    end

    M = A[1:size, 1:size]

    prev_pivot = 1
    for k in 1:size
        current_pivot = M[k, k]
        if current_pivot == 0
            return false
        end

        if k < size
            for i in (k + 1):size
                for j in (k + 1):size
                    M[i, j] = div(M[k, k] * M[i, j] - M[i, k] * M[k, j], prev_pivot)
                end
            end
            prev_pivot = current_pivot
        end
    end
    return true
end

const counts_filename = "lu_counts_$(N)x$(N).json"
const MASK_BITS = SUBMATRIX_SIZE * SUBMATRIX_SIZE
const NUM_BASE_MATRICES = 1 << (MASK_BITS - 1)

counts_data = Dict{String, Int64}()

println("Starting search...")

overall_start_time = time()

for i in 0:(NUM_BASE_MATRICES - 1)
    mask_4x4 = (i << 1) | 1

    # Construct the base (N-1)x(N-1) matrix
    A_base = zeros(Int64, SUBMATRIX_SIZE, SUBMATRIX_SIZE)
    bit_pos = 0
    for r in 1:SUBMATRIX_SIZE
        for c in 1:SUBMATRIX_SIZE
            A_base[r, c] = (mask_4x4 >> bit_pos) & 1
            bit_pos += 1
        end
    end

    if can_lu_factorize(A_base, SUBMATRIX_SIZE)
        A = zeros(Int64, N, N)
        A[1:SUBMATRIX_SIZE, 1:SUBMATRIX_SIZE] = A_base
        
        local_stats = Dict{String, Int64}()
        num_remaining_combinations = 1 << (N*N - SUBMATRIX_SIZE*SUBMATRIX_SIZE)

        # For each factorizable (N-1)x(N-1) matrix, iterate over all combinations
        # of the remaining entries in the NxN matrix
        for j in 0:(num_remaining_combinations - 1)
            current_bit = 0
            for r in 1:N
                for c in 1:N
                    if r > SUBMATRIX_SIZE || c > SUBMATRIX_SIZE
                        A[r, c] = (j >> current_bit) & 1
                        current_bit += 1
                    end
                end
            end

            if can_lu_factorize(A, N)
                row_sums = sum(A, dims=2)[:]
                col_sums = sum(A, dims=1)[:]
                sort!(row_sums)
                sort!(col_sums)

                key = join(row_sums, " ") * "," * join(col_sums, " ")
                local_stats[key] = get(local_stats, key, 0) + 1
            end
        end

        for (key, val) in local_stats
            counts_data[key] = get(counts_data, key, 0) + val
        end
    end
end

println("\nSaving final results...")
open(counts_filename, "w") do f
    JSON.print(f, counts_data, 4)
end

overall_end_time = time()
total_time_sec = overall_end_time - overall_start_time

total_matrices_found = 0
if !isempty(counts_data)
    total_matrices_found = sum(values(counts_data))
end

println("\n-------------------------------------------------")
println("Search complete.")
@printf("Total valid 6x6 matrices found: %d\n", total_matrices_found)
@printf("Total unique row/col sum profiles: %d\n", length(counts_data))
@printf("Total time taken: %.2f minutes\n", total_time_sec / 60.0)
println("Results saved to: $counts_filename")
println("-------------------------------------------------")