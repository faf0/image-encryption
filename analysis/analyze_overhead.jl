using PyPlot

if length(ARGS) != 1
  error("must provide date as argument")
end

global const date = ARGS[1]
global const csv_file = string("../results/", date, "-overhead.csv")

no_tests = 20
(data, header) = readdlm(csv_file, ';', has_header=true)
rows = size(data)[1]
cols = size(data)[2]
no_widths = div(rows, no_tests)
row_sel = Array(typeof(1:1), no_widths)
means = Array(Float64, (no_widths, cols))
stds = Array(Float64, (no_widths, cols))
toidx = Dict{ASCIIString, typeof(1)}()

for i=1:cols
    push!(toidx, header[i], i)
end

for i=1:no_widths
    row_sel[i] = ((i - 1) * 20 + 1):(i * 20)
    means[i,:] = mean(data[row_sel[i], :], 1)
    stds[i,:] = std(data[row_sel[i], :], 1)
end

