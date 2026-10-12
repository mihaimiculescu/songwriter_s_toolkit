def parabolic_interpolation(a, b, c):
    """
    Literal translation of eckf_pitch_final/parabolic_interpolation.m.

    MATLAB:
        pos = 0.5 * ((a-c)/(a - 2*b + c));
        peak = b - 0.25*(a-c)*pos;
    """
    denominator = a - 2 * b + c
    pos = 0.5 * ((a - c) / denominator)
    peak = b - 0.25 * (a - c) * pos
    return peak, pos
