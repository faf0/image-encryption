using ColorImageEncryption

import ColorImageEncryption.pwlcm

# Applies Chen-generated random values to color channels of pixel (encryption
# direction).
function applychenpixel(r::Uint8, g::Uint8, b::Uint8,
                        rp::Uint8, gp::Uint8, bp::Uint8,
                        xp::Uint8, yp::Uint8, zp::Uint8)
  rpn = applychenbyte(r, rp, xp)
  gpn = applychenbyte(g, gp, yp)
  bpn = applychenbyte(b, bp, zp)
  return (rpn, gpn, bpn)
end

function unequal(x, y)
  if x != y
    error(x, " != ", y)
  end
end

# one-based bit extraction from unsigned byte (Uint8).
# i = 1 returns least significant bit
# i = 8 returns most significant bit
# returns either 0 or 1 with typeof Uint8
function extractbit(b::Uint8, i)
  mask = 0x01 << (i - 1)
  if (b & mask) != 0x00
    return one(Uint8)
  else
    return zero(Uint8)
  end
end

# return true if the ith bit of the given integer b is set.
# i = 1 is the least significant bit.
# returns false, otherwise.
function extractbitgeneral(b, i)
  return (div(b, 2^(i - 1)) % 2) != 0
end

# Returns b1 where the bit at position pos (one-based) is replaced with the
# corresponding bit from b2
function setbit(b1::Uint8, b2::Uint8, pos)
  mask = 0x01 << (pos - 1)
  btmp = b2 & mask
  return (btmp == 0x00) ? b1 & (~mask) :  b1 | btmp
end

# Compute setbit with function and using different method.
# Throw error, if results do not match.
function testsetbit(runs)
  for i=1:runs
     x = mod(ifloor(rand(1) * 256)[1], 256)
     x = convert(Uint8, x)
     y = mod(ifloor(rand(1) * 256)[1], 256)
     y = convert(Uint8, y)
     pos = (ifloor(rand(1) * 8))[1] + 1
     xs = setbit(x, y, pos)
     mask = 0x01 << (pos - 1)
     xset = (x & mask) != 0x00
     yset = (y & mask) != 0x00
     if yset
       if xset
         unequal(xs, x)
       else
         unequal(xs, x + mask)
       end
     else
        if xset
          unequal(xs, x - mask)
        else
          unequal(xs, x)
        end
     end
  end
end

# test PWLCM
x = 0.2345
for i=1:100
  x = pwlcm(x, 0.1234)
  println(x)
end
testsetbit(512)
