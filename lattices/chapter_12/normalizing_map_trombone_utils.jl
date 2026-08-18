using LinearAlgebra: inv

# Phase trombone utilities using SciBmad's transverse normalizing matrix `N`.
# SciBmad 0.5 returns N as a 4x4 matrix. The trombone acts on the transverse
# coordinates and passes z and pz through unchanged.

function load_ring(path; make_trombones=false, kwargs...)
    ring = include(path)
    if make_trombones
        make_trombones!(ring; kwargs...)
    end
    return ring
end

function Trombone(; dnu1=0.0, dnu2=0.0, dnu3=0.0, kwargs...)
    return LineElement(;
        kind="Trombone",
        L=0.0,
        transport_map_params=(dnu1, dnu2, dnu3),
        kwargs...,
    )
end

function _normalizing_map_property(x, name, default=nothing)
    hasproperty(x, name) || return default
    try
        return getproperty(x, name)
    catch
        return default
    end
end

function _normalizing_map_trombone_element_index(element, ring)
    for (idx, candidate) in enumerate(ring.line)
        candidate === element && return idx
    end

    element_name = _normalizing_map_property(element, :name)
    if element_name !== nothing
        row = findfirst(
            candidate -> _normalizing_map_property(candidate, :name) == element_name,
            ring.line,
        )
        row !== nothing && return row
    end

    error("Could not find trombone element in ring.line.")
end

function _normalizing_map_trombone_ring_element(element, ring)
    return ring.line[_normalizing_map_trombone_element_index(element, ring)]
end

function _normalizing_map_trombone_twiss_row(element, ring, twiss_table)
    idx = _normalizing_map_trombone_element_index(element, ring)

    hasproperty(twiss_table, :index) || error("Twiss DataFrame has no index column.")
    row = findfirst(==(idx), twiss_table.index)
    row !== nothing && return row
    error("Trombone element is missing from the Twiss DataFrame.")
end

function _normalizing_map_twiss(ring; twiss_kwargs...)
    return twiss(ring; cols=["N"], twiss_kwargs...)
end

function _normalizing_map_from_table(twiss_table, row)
    if hasproperty(twiss_table, :N)
        return twiss_table.N[row]
    end

    error(
        "Twiss table has no `N` column. Compute twiss with cols=[\"N\"]."
    )
end

function _normalizing_map_linear_matrix(N)
    if N isa AbstractMatrix
        return Matrix(N)
    end

    try
        return Matrix(jacobian(N, SciBmad.NNF.VARS_CPARAM))
    catch
        try
            return Matrix(jacobian(N))
        catch err
            error("Could not convert the normalizing matrix `N` into a 4x4 matrix: $err")
        end
    end
end

function _normalizing_map_check(A)
    size(A) == (4, 4) || error("Normalizing-map trombone expects SciBmad's 4x4 N matrix.")
    return nothing
end

function normalizing_map_trombone_rotation(dphi_1, dphi_2, dphi_3, sample; sin_sign=1)
    iszero(dphi_3) || error("SciBmad's N column is transverse; dphi_3 must be zero.")
    c1 = cos(dphi_1)
    s1 = sin_sign * sin(dphi_1)
    c2 = cos(dphi_2)
    s2 = sin_sign * sin(dphi_2)

    one_entry = one(sample) * one(c1)
    zero_entry = zero(one_entry)
    R = fill(zero_entry, 4, 4)

    for i in 1:4
        R[i, i] = one_entry
    end

    R[1, 1] = c1
    R[1, 2] = s1
    R[2, 1] = -s1
    R[2, 2] = c1

    R[3, 3] = c2
    R[3, 4] = s2
    R[4, 3] = -s2
    R[4, 4] = c2

    return R
end

function normalizing_map_trombone_matrix(A::AbstractMatrix, dphi_1, dphi_2, dphi_3=0.0; sin_sign=1)
    _normalizing_map_check(A)
    R = normalizing_map_trombone_rotation(dphi_1, dphi_2, dphi_3, A[1, 1]; sin_sign=sin_sign)
    return A * R * inv(A)
end

function normalizing_map_trombone_matrix(A::AbstractMatrix, Ainv::AbstractMatrix, dphi_1, dphi_2, dphi_3=0.0; sin_sign=1)
    _normalizing_map_check(A)
    _normalizing_map_check(Ainv)
    R = normalizing_map_trombone_rotation(dphi_1, dphi_2, dphi_3, A[1, 1]; sin_sign=sin_sign)
    return A * R * Ainv
end

function _normalizing_map_trombone_apply(M, v)
    return (
        M[1, 1] * v[1] + M[1, 2] * v[2] + M[1, 3] * v[3] + M[1, 4] * v[4],
        M[2, 1] * v[1] + M[2, 2] * v[2] + M[2, 3] * v[3] + M[2, 4] * v[4],
        M[3, 1] * v[1] + M[3, 2] * v[2] + M[3, 3] * v[3] + M[3, 4] * v[4],
        M[4, 1] * v[1] + M[4, 2] * v[2] + M[4, 3] * v[3] + M[4, 4] * v[4],
        v[5],
        v[6],
    )
end

function normalizing_map_trombone_map(A; sin_sign=1)
    _normalizing_map_check(A)
    Ainv = inv(A)

    return function (v, q, params)
        dnu1, dnu2, dnu3 = _normalizing_map_trombone_param_tuple(params)
        M = normalizing_map_trombone_matrix(A, Ainv, dnu1, dnu2, dnu3; sin_sign=sin_sign)
        return (_normalizing_map_trombone_apply(M, v), q)
    end
end

function normalizing_map_trombone_A(element, ring, twiss_table)
    row = _normalizing_map_trombone_twiss_row(element, ring, twiss_table)
    A = _normalizing_map_linear_matrix(_normalizing_map_from_table(twiss_table, row))
    _normalizing_map_check(A)
    return A
end

function normalizing_map_trombone_params(dnu1, dnu2, dnu3=0.0)
    return (dnu1, dnu2, dnu3)
end

function _normalizing_map_trombone_param_tuple(params)
    if length(params) == 2
        return (params[1], params[2], 0.0)
    elseif length(params) == 3
        return (params[1], params[2], params[3])
    end

    error("Trombone expects two or three transport_map_params: (dnu1, dnu2[, dnu3]).")
end

function normalizing_map_trombone_params(element)
    params = _normalizing_map_property(element, :transport_map_params)
    params === nothing && error("Trombone element has no transport_map_params. Expected (dnu1, dnu2[, dnu3]).")
    return _normalizing_map_trombone_param_tuple(params)
end

function mark_trombone!(element, dnu1, dnu2; dnu3=0.0)
    element.kind = "Trombone"
    element.transport_map_params = normalizing_map_trombone_params(dnu1, dnu2, dnu3)
    return element
end

function mark_trombone!(element, ring, dnu1, dnu2; dnu3=0.0)
    mark_trombone!(element, dnu1, dnu2; dnu3=dnu3)

    idx = _normalizing_map_trombone_element_index(element, ring)
    ring_element = ring.line[idx]
    mark_trombone!(ring_element, dnu1, dnu2; dnu3=dnu3)
    return ring_element
end

function normalizing_map_trombone_elements(ring)
    return filter(element -> _normalizing_map_property(element, :kind) == "Trombone", ring.line)
end

function _set_normalizing_map_trombone!(element, A, dnu1, dnu2, dnu3=0.0; sin_sign=1)
    element.transport_map = normalizing_map_trombone_map(A; sin_sign=sin_sign)
    element.transport_map_params = normalizing_map_trombone_params(dnu1, dnu2, dnu3)
    return element
end

function make_trombone!(element, ring, twiss_table, dnu1, dnu2; dnu3=0.0, sin_sign=1)
    ring_element = _normalizing_map_trombone_ring_element(element, ring)
    A = normalizing_map_trombone_A(ring_element, ring, twiss_table)
    _set_normalizing_map_trombone!(ring_element, A, dnu1, dnu2, dnu3; sin_sign=sin_sign)

    if ring_element !== element
        _set_normalizing_map_trombone!(element, A, dnu1, dnu2, dnu3; sin_sign=sin_sign)
    end

    return ring_element
end

function make_trombone!(element, ring, twiss_table; sin_sign=1)
    dnu1, dnu2, dnu3 = normalizing_map_trombone_params(element)
    return make_trombone!(element, ring, twiss_table, dnu1, dnu2; dnu3=dnu3, sin_sign=sin_sign)
end

function make_trombone!(element, ring; sin_sign=1, twiss_kwargs...)
    tw = _normalizing_map_twiss(ring; twiss_kwargs...)
    return make_trombone!(element, ring, tw.df; sin_sign=sin_sign)
end

function _make_trombone_from_spec!(ring, twiss_table, element; sin_sign=1)
    return make_trombone!(element, ring, twiss_table; sin_sign=sin_sign)
end

function _make_trombone_from_spec!(ring, twiss_table, spec::Tuple; sin_sign=1)
    if length(spec) == 3
        element, dnu1, dnu2 = spec
        return make_trombone!(element, ring, twiss_table, dnu1, dnu2; sin_sign=sin_sign)
    elseif length(spec) == 4
        element, dnu1, dnu2, dnu3 = spec
        return make_trombone!(element, ring, twiss_table, dnu1, dnu2; dnu3=dnu3, sin_sign=sin_sign)
    end

    error("Trombone spec tuple must be (element, dnu1, dnu2[, dnu3]).")
end

function make_trombones!(ring, trombones; sin_sign=1, twiss_kwargs...)
    tw = _normalizing_map_twiss(ring; twiss_kwargs...)

    for trombone in trombones
        _make_trombone_from_spec!(ring, tw.df, trombone; sin_sign=sin_sign)
    end

    return ring
end

function make_trombones!(ring; sin_sign=1, twiss_kwargs...)
    return make_trombones!(ring, normalizing_map_trombone_elements(ring); sin_sign=sin_sign, twiss_kwargs...)
end
