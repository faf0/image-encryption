module ColorImageEncryption

using CUDA
using Images
#using ImageView

export pwlcm, encrypt, decrypt, test

global const debug = false
global const gpu = false
global const path = "./"

# generates the next element in the piecewise-linear chaotic map sequence.
# 0 <= x < 1
# 0 < p < 0.5
function pwlcm(x, p)
  if 0 <= x < p
    return x / p
  elseif p <= x < 0.5
    return (x - p) / (0.5 - p)
  else
    # 0.5 <= x < 1
    return pwlcm(1 - x, p)
  end
end

# permutes the given array in-place.
# provided that the PWLCM produces a pseudo-random sequence,
# the result is uniform random permutation according to p. 126 of CLRS, ed. 3.
function permarray(len, x, p)
  array = zeros(Uint32, len)
  array[1:] = 1:len
  xi = x
  for i=len:-1:1
    xi = pwlcm(xi, p)
    index = mod(ifloor(xi * len), i) + 1
    tmp = array[i]
    array[i] = array[index]
    array[index] = tmp
  end
  return array
end

# Maps a column from M x (N x 24) bit-space to M x (N x 3) byte-space.
# The memory looks as follows (indices refer to bits):
# [1, 2, ..., 8N] ~ Red Channel
# [8N + 1, ..., 16N] ~ Green Channel
# [16N + 1, ..., 24N] ~ Blue Channel
# We have 3 color channels.
function mapcol(col, len)
   # len = 3 * 8 * N
   # Thus, len / 3 = 8N is the width of a color channel
   newch = div(col - 1, div(len, 3)) + 1
   # We have len / 3 = 8N bits per color channel
   # 8 bits are in a byte
   newbyte = div(mod(col - 1, div(len, 3)), 8) + 1
   # We have len / 24 = N bytes per color channel
   #newbyte = mod(col - 1, div(len, 24)) + 1
   # We have 8 bits in a byte
   newbit = mod(col - 1, 8) + 1
   #newbit = mod(newbyte - 1, 8) + 1
   return (newch, newbyte, newbit)
end

# Returns byte b1 where the bit at position b1pos (1 <= b1pos <= 8; 1 is least
# significant bit) is replaced with the bit at b2pos (same conventions as for
# b1pos) from byte b2.
function setbitbranch(b1::Uint8, b2::Uint8, b1pos, b2pos)
  # extract bit from b2
  mask = 0x01 << (b2pos - 1)
  btmp = b2 & mask
  # set bit in b1
  mask = 0x01 << (b1pos - 1)
  if btmp == 0x00
     # bit of b2 at b2pos is zero. set bit of b1 at b1pos to zero.
     return b1 & (~mask)
  else
     # bit of b2 at b2pos is one. make sure that bit at b1pos is one.
     return b1 | mask
  end
end

# Returns byte b1 where the bit at position b1pos (1 <= b1pos <= 8; 1 is least
# significant bit) is replaced with the bit at b2pos (same conventions as for
# b1pos) from byte b2.
function setbit(b1::Uint8, b2::Uint8, b1pos, b2pos)
  # extract most significant part of b1
  m = (b1 >> b1pos) << b1pos
  # extract bit from b2 and shift it to b1pos
  b2shift = ((b2 >> (b2pos - 1)) & 0x01) << (b1pos - 1)
  # extract least significant part of b1
  l = b1 & ((0x01 << (b1pos - 1)) - 1)
  # combine parts
  return m | b2shift | l
end

# Sets the given old bit from arrayIn and write it to the newbit in arrayOut
function setBitInPixel(arrayIn, arrayOut, len, rowold, colold, rownew, colnew)
  (oldch, cololdbyte, oldbit) = mapcol(colold, len)
  (newch, colnewbyte, newbit) = mapcol(colnew, len)
  if debug
    println("old (ch, byte, bit): ", oldch, " ", cololdbyte, " ", oldbit)
    println("new (ch, byte, bit): ", newch, " ", colnewbyte, " ", newbit)
  end

  # setbit must not modify newbyte!
  # Otherwise, temporary variable needed.
  oldbyte = arrayIn[rowold, cololdbyte, oldch]
  newbyte = arrayOut[rownew, colnewbyte, newch]
  if debug
    println("old/new byte:")
    println(bits(oldbyte))
    println(bits(newbyte))
  end

  arrayOut[rownew, colnewbyte, newch] = setbit(newbyte, oldbyte, newbit, oldbit)

  if debug
    println("mod new byte:")
    println(bits(arrayOut[rownew, colnewbyte, newch]))
    println()
  end
end

function gpupermute(imA, imE, H, V)
  try
    dev = CuDevice(0)
    ctx = create_context(dev)
    md = CuModule("perm.ptx")
    # load permute function from ptx file
    f = CuFunction(md, "permute")

    M = size(imA)[1]
    N = size(imA)[2]
    len = M * N

    r = imA[1:, 1:, 1]
    g = imA[1:, 1:, 2]
    b = imA[1:, 1:, 3]

    gr = CuArray(r)
    gg = CuArray(g)
    gb = CuArray(b)

    goutr = CuArray(Uint8, len)
    goutg = CuArray(Uint8, len)
    goutb = CuArray(Uint8, len)

    gH = CuArray(H)
    gV = CuArray(V)

    # launch parameters:
    #  function, grid dim (blocks), block dim (threads),
    #  tuple of function parameters
    launch(f, M, N, (gr, gg, gb, goutr, goutg, goutb, gH, gV,
      convert(Uint32, M), convert(Uint32, N)))

    free(gr)
    free(gg)
    free(gb)

    free(gH)
    free(gV)

    hr = to_host(goutr)
    hg = to_host(goutg)
    hb = to_host(goutb)

    free(goutr)
    free(goutg)
    free(goutb)

    display(hr[1:10])
    imE[1:, 1:, 1] = hr
    imE[1:, 1:, 2] = hg
    imE[1:, 1:, 3] = hb

    unload(md)
    destroy(ctx)
  catch err
    if isa(err, CuDriverError)
      println("$err: $(description(err))")
    else
      throw(err)
    end
  end
end

# Stores the permutation (part of encryption) of imA in imE.
# H defines the row permutation and V defines the column permutation.
function permute(imA, imE, H, V)
  if debug
    for row=1:1
      for col=1:8
        setBitInPixel(imA, imE, length(V), row, col, H[row], V[col])
      end
    end
  else
     for row=1:length(H)
       for col=1:length(V)
         setBitInPixel(imA, imE, length(V), row, col, H[row], V[col])
       end
     end
  end
end

# Stores the reverse permutation (part of decryption) of imE in imD.
# H defines the reverse row permutation and V defines the reverse
# column permutation.
function reversepermute(imE, imD, H, V)
    if debug
      for row=1:1
        for col=1:8
          setBitInPixel(imE, imD, length(V), H[row], V[col], row, col)
        end
      end
    else
      for row=1:length(H)
        for col=1:length(V)
          setBitInPixel(imE, imD, length(V), H[row], V[col], row, col)
        end
      end
    end
end

# Creates the row and column permutation arrays.
# H is the row permutation array.
# V is the column-bit permutation array.
function permarrays(key, M, N)
    H = permarray(M, key["x0"], key["px"])
    V = permarray(N * 24, key["y0"], key["py"])
    permuted = (sort(H) == 1:M) && (sort(V) == 1:(24*N))
    if !permuted
      error("H or V are not permutations!")
    end
    return (H, V)
end

# Returns the width M and height N of the image.
function imgparams(imA)
    imAsize = size(imA)
    M = imAsize[1]
    N = imAsize[2]
    return (M, N)
end

# Applies the Chen system on the given three-tuple.
# Returns the transformed three-tuple.
# The following conditions must hold in order for the Chen system to be chaotic:
# a = 35
# b = 3
# 20 <= c <= 28.4
function chen(x::FloatingPoint, y::FloatingPoint, z::FloatingPoint, c=20)
  const a = 35
  const b = 3
  # prevent floating point overflows leading to NaNs
  # FIXME mod 10^8 operations not in original algorithm!
  x = mod(x, 10^8)
  y = mod(y, 10^8)
  z = mod(z, 10^8)
  xp = a * (y - x)
  yp = (c - a) * x - x * z + c * y
  zp = x * y - b * z
  return (xp, yp, zp)
end

# Maps floating-point numbers generated by Chen system to byte space[0, 255].
# Turns number in Chen number sequence to byte in S_X, S_Y, S_Z.
function chentobyte(x::FloatingPoint)
  xp = abs(x)
  xbyte = mod((xp - floor(xp)) * 1.0e14, 256)
  return convert(Uint8, ifloor(xbyte))
end

# Initializes the Chen system.
# Executes key["n0"] iterations and returns the (x, y, z) vector (not sequence
# (x, y, z) vectors!
function cheninit(key)
  x = key["xc0"]
  y = key["yc0"]
  z = key["zc0"]

  # iteratate Chen system n0 times
  for i=1:key["n0"]
      (x, y, z) = chen(x, y, z, key["c"])
  end

  return (x, y, z)
end

# Generates the next Chen byte sequence based on the given x, y, z.
# There is one sequence per color channel (red, green, and blue).
# Also, returns the latest (x, y, z) vector.
function chenbyteseqnext(MN, key, x, y, z)
  SX = Array(Uint8, MN)
  SY = Array(Uint8, MN)
  SZ = Array(Uint8, MN)

  # generate Chen-based byte sequence of length MN
  for i=1:MN
      (x, y, z) = chen(x, y, z, key["c"])
      SX[i] = chentobyte(x)
      SY[i] = chentobyte(y)
      SZ[i] = chentobyte(z)
   end

   return (SX, SY, SZ, x, y, z)
end

# Creates the first byte sequences, i.e., initalizes the (x, y, z) vector
# and actually procuces the first sequence of (x, y, z) vectors.
# Returns the latest (x, y, z) vector.
function chenbyteseqfirst(MN, key)
  (x, y, z) = cheninit(key)
  (SX, SY, SZ, x, y, z) = chenbyteseqnext(MN, key, x, y, z)
  return (SX, SY, SZ, x, y, z)
end

# Applies a Chen-generated random value to the color channel of a pixel.
# c is the current pixel
# cp is the previous pixel (left neighbor) pixel
# s is the value from the sequence
function applychenbyte(c::Uint8, cp::Uint8, s::Uint8)
   # Julia converts c + cp to Uint64
   return convert(Uint8, mod(c + cp, 256)) $ s
end

# Undos the Chen system pixel transformation (decryption direction).
# c is the current pixel
# cp is the previous (left neighbor) pixel
# s is the value from the sequence
function undochenbyte(c::Uint8, cp::Uint8, s::Uint8)
  # the result of the subtraction is a Uint64
  return convert(Uint8, mod((c $ s) - cp, 256))
end

# Applies the given sequences on the given image.
# enc: true for encryption direction, false for decryption direction.
function applychenimage(enc, im, key, SX, SY, SZ)
  (M, N) = imgparams(im)
  # left neighbor neighbor value
  pre = key["rp0"]
  S = SX
  for ch=1:3
    if ch == 1
        pre = key["rp0"]
        S = SX
    elseif ch == 2
        pre = key["gp0"]
        S = SY
    else
        pre = key["bp0"]
        S = SZ
    end
    for row=1:M
      for col=1:N
        tmp = im[row, col, ch]
        if enc
          new_byte = applychenbyte(tmp, pre, S[(row - 1) * N + col])
          im[row, col, ch] = new_byte
          pre = new_byte
        else
          new_byte = undochenbyte(tmp, pre, S[(row - 1) * N + col])
          im[row, col, ch] = new_byte
          pre = tmp
        end
      end
    end
  end
end

# Encrypts the given input file and writes the result to the output file
function encrypt(fileIn, fileOut, key)
    # LOAD PLAIN IMAGE
    img = imread(fileIn)
    imA = convert(Array, img)
    #imAR = imA[1:, 1:, 1]
    #imAG = imA[1:, 1:, 2]
    #imAB = imA[1:, 1:, 3]
    #imG = grayim(img)
    # see https://github.com/timholy/Images.jl/blob/master/doc/core.md
    (M, N) = imgparams(imA)
    (H, V) = permarrays(key, M, N)

    imE = Array(typeof(imA[1]), size(imA))
    # slower alternatives:
    #imE = zeros(typeof(imA[1], size(imA))
    #imE = copy(imA)

    # PERMUTATION
    if gpu
      gpupermute(imA,imE, H, V)
    else
      permute(imA, imE, H, V)
    end

    # CHEN SYSTEM TRANSFORMATION
    MN = M * N
    (SX, SY, SZ, x, y, z) = chenbyteseqfirst(MN, key)
    applychenimage(true, imE, key, SX, SY, SZ)

    for i=2:key["beta"]
        (SX, SY, SZ, x, y, z) = chenbyteseqnext(MN, key, x, y, z)
        applychenimage(true, imE, key, SX, SY, SZ)
    end

    # OUTPUT IMAGE
    #ImageView.display(imE)
    imwrite(imE, fileOut)
    return imE
end

# Decrypts the given input file and writes the result to the output file
function decrypt(fileIn, fileOut, key)
    # LOAD ENCRYPTED IMAGE
    img = imread(fileIn)
    imE = convert(Array, img)
    imD = Array(typeof(imE[1]), size(imE))
    (M, N) = imgparams(imE)
    (H, V) = permarrays(key, M, N)
    # create the sequence lists first and store them in an array of lists (of
    # sequences which are represented as arrays)
    chenbyteseqs = Array((Array, Array, Array), key["beta"])

    # UNDO CHEN SYSTEM TRANSFORMATION
    MN = M * N
    (SX, SY, SZ, x, y, z) = chenbyteseqfirst(MN, key)
    chenbyteseqs[1] = (SX, SY, SZ)

    for i=2:key["beta"]
       (SX, SY, SZ, x, y, z) = chenbyteseqnext(MN, key, x, y, z)
       chenbyteseqs[i] = (SX, SY, SZ)
    end

    for i=key["beta"]:-1:1
      (SX, SY, SZ) = chenbyteseqs[i]
      applychenimage(false, imE, key, SX, SY, SZ)
    end

    # UNDO PERMUTATION
    reversepermute(imE, imD, H, V)

    # OUTPUT IMAGE
    #ImageView.display(imD)
    imwrite(imD, fileOut)
    return imD
end

function test()
    # Key constraints
    # x0 (same for y0): 0 <= x0 < 1
    # px (same for py): 0 < px < 0.5
    # 2 <= beta <= 8
    key = [ ### secret key initial values ###
            # PWLCM initial values
           "x0" => 0.6789, "y0" => 0.3456,
            # Chen system initial values
           "xc0" => -10.21829319252345, "yc0" => 0.98182971820392,
             "zc0" => 37.8736176298736,
            # Chen system pixel application initial values
           "rp0" => convert(Uint8, 123),
           "gp0" => convert(Uint8, 12),
           "bp0" => convert(Uint8, 23),
           ### Parameters ###
            # PWLCM parameters
           "px" => 0.2345, "py" => 0.1234,
            # Chen system parameter
            "c" => 28,
            # Iteration times for Chen system
           "n0" => 189,
           "beta" => 2 ]
    img = imread(string(path, "lenna.png"))
    imA = convert(Array, img)
    imE = encrypt(string(path, "lenna.png"), string(path, "lenna-e.png"), key)
    imD = decrypt(string(path, "lenna-e.png"), string(path, "lenna-d.png"), key)

    if imA != imD
      error("decrypted image differs from original image!")
    end
end

test()

end # module ColorImageEncryption
