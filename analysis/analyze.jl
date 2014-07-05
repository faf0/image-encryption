using PyPlot

if length(ARGS) != 1
  error("must provide date as argument")
end

global const date = ARGS[1]
global const csv_file = string("../results/", date, "-results.csv")
global const plotdir = "../plots/"

function draw_plot(x, cpuy, gpuy, title, filepath)
    spdup = cpuy ./ gpuy

    figure(figsize=(10, 8))
    #figure()

    subplot(211)
    plot(x, cpuy, "go-", label="CPU")
    plot(x, gpuy, "bo-", label="GPU")
    legend(loc="upper left")
    xticks(x)
    ylabel("Execution Time [ms]")

    subplot(212)
    plot(x, spdup, "ro-", label="Speedup")
    #legend(loc="upper left")
    xticks(x)
    ylabel("Speedup (GPU / CPU)")
    xlabel("Image Size [n x n]")

    suptitle(title)

    savefig(filepath)
end

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

x = means[:, toidx["img_width"]]

cpuy = means[:, toidx["cpu_enc_perm"]]
gpuy = means[:, toidx["gpu_enc_perm"]]

draw_plot(x, cpuy, gpuy, "Permutation (Encryption)", string(plotdir, "time-enc-perm.pdf"))

cpuy = means[:, toidx["cpu_enc_chen"]]
gpuy = means[:, toidx["gpu_enc_chen"]] ./ means[:, toidx["no_imgs_gpu_enc_chen"]]

draw_plot(x, cpuy, gpuy, string("Chen Transform (Encryption, ",
  convert(typeof(1), means[1, toidx["no_imgs_gpu_enc_chen"]]), " images)"),
  string(plotdir, "time-enc-chen.pdf"))

cpuy = means[:, toidx["cpu_dec_chen"]]
gpuy = means[:, toidx["gpu_dec_chen"]]

draw_plot(x, cpuy, gpuy, "Chen Transform (Decryption)", string(plotdir, "time-dec-chen.pdf"))

cpuy = means[:, toidx["cpu_dec_perm"]]
gpuy = means[:, toidx["gpu_dec_perm"]]

draw_plot(x, cpuy, gpuy, "Permutation (Decryption)", string(plotdir, "time-dec-perm.pdf"))

