using RandomLotkaVolterraCavity  # bring package under test into scope
using Test  # base testing utilities
using Aqua  # run static package checks
using Random  # deterministic RNG fixtures
using Distributions  # probability distributions used by models
using StatsBase  # sampling helpers for discrete distributions

const BASE_SEED = UInt32(0x12345678)  # global seed so RNG-driven tests are reproducible

# helper that produces a simple chain graph adjacency matrix
simple_adj(N::Int, K; rng)::Matrix{Float64} = begin
    adj = zeros(Float64, N, N)
    for i in 1:(N - 1)
        adj[i, i + 1] = adj[i + 1, i] = 1.0  # undirected nearest-neighbour links
    end
    adj
end

const DEGREES = DiscreteNonParametric([0, 1, 2], [0.1, 0.6, 0.3])  # degree law for population dynamics
const CAVITY_DEGREES = DiscreteNonParametric([0, 1, 2], [0.2, 0.5, 0.3])  # cavity degree distribution
const P0_DIST = Exponential(1.0)  # immigration-dominated initial condition sampler

base_J = [0.0 0.2; -0.1 0.0]  # baseline asymmetric coupling matrix for direct Model tests
base_model = Model(2, 4, base_J, 0.05, P0_DIST, Deterministic())  # deterministic model on fixed graph
base_model_dis = ModelDisordered(2, 1.0, 4, simple_adj, DEGREES, CAVITY_DEGREES, 0.2, 0.3, 0.1, 0.05, P0_DIST, Deterministic())  # disordered sparse model fixture
base_model_dis_fc = ModelDisorderedFC(2, 4, 0.2, 0.3, 0.1, 0.05, P0_DIST, Deterministic())  # fully-connected disordered fixture

@testset "RandomLotkaVolterraCavity" begin
    @test base_model.N == 2  # structural sanity check on stored fields
    @test base_model.M == 4
    @test base_model.lam == 0.05
    @test base_model.noise isa Deterministic
    @test size(base_model.J) == (2, 2)  # confirm coupling matrix retained

    @test base_model_dis.N == 2  # disordered model field checks
    @test base_model_dis.K == 1.0
    @test base_model_dis.M == 4
    @test base_model_dis.noise isa Deterministic
    @test base_model_dis.m == 0.2
    @test base_model_dis.sigma2 == 0.3
    @test base_model_dis.corr == 0.1
    @test base_model_dis.lam == 0.05
    @test base_model_dis.p0 isa Exponential
    @test base_model_dis.generate_adj === simple_adj  # confirm function handle retained
    @test size(base_model_dis.generate_adj(base_model_dis.N, 1.0; rng=Xoshiro(BASE_SEED))) == (2, 2)  # confirm adjacency generation

    @test base_model_dis_fc.N == 2  # fully-connected disordered model field checks
    @test base_model_dis_fc.M == 4
    @test base_model_dis_fc.noise isa Deterministic
    @test base_model_dis_fc.m == 0.2
    @test base_model_dis_fc.sigma2 == 0.3
    @test base_model_dis_fc.corr == 0.1
    @test base_model_dis_fc.lam == 0.05
    @test base_model_dis_fc.p0 isa Exponential

    @testset "Cavity and Marginal messages" begin
        rng = Xoshiro(BASE_SEED)  # consistent RNG per sub-test
        rng_ref = copy(rng)  # reference generator to reproduce random draws

        cav_zero = CavityFP(1, 2, :zero, 0.0, 0.0, 0.0, rng)  # zero initialisation
        @test cav_zero.i == 1
        @test cav_zero.j == 2
        @test cav_zero.mu == 0.0
        @test cav_zero.q == 0.0
        @test cav_zero.chi == 0.0
        
        cav_rand = CavityFP(1, 2, :random, 0.0, 0.0, 0.0, rng)  # random initialisation path
        @test cav_rand.i == 1
        @test cav_rand.j == 2
        @test cav_rand.mu == rand(rng_ref)
        @test cav_rand.q == rand(rng_ref)
        @test cav_rand.chi == 0.0

        cav_custom = CavityFP(1, 2, :custom, 0.4, 0.2, 0.1, Xoshiro(BASE_SEED))  # custom seeds propagate as-is
        @test cav_custom.i == 1
        @test cav_custom.j == 2
        @test cav_custom.mu == 0.4
        @test cav_custom.q == 0.2
        @test cav_custom.chi == 0.1

        marg_zero = MarginalFP(1, :zero, 0.0, 0.0, 0.0, rng)
        @test marg_zero.i == 1
        @test marg_zero.mu == 0.0
        @test marg_zero.q == 0.0
        @test marg_zero.chi == 0.0

        marg_rand = MarginalFP(1, :random, 0.0, 0.0, 0.0, rng)
        @test marg_rand.i == 1
        @test marg_rand.mu == rand(rng_ref)
        @test marg_rand.q == rand(rng_ref)
        @test marg_rand.chi == 0.0

        marg_custom = MarginalFP(1, :custom, 0.4, 0.2, 0.1, Xoshiro(BASE_SEED))
        @test marg_custom.i == 1
        @test marg_custom.mu == 0.4
        @test marg_custom.q == 0.2
        @test marg_custom.chi == 0.1

        cav_q0 = CavityFP_q0(1, 2, :zero, 0.0, rng)
        @test cav_q0.i == 1
        @test cav_q0.j == 2
        @test cav_q0.x == 0.0

        cav_q0_rand = CavityFP_q0(1, 2, :random, 0.0, rng)
        @test cav_q0_rand.i == 1
        @test cav_q0_rand.j == 2
        @test cav_q0_rand.x == rand(rng_ref)

        cav_q0_custom = CavityFP_q0(1, 2, :custom, 0.8, Xoshiro(BASE_SEED))
        @test cav_q0_custom.i == 1
        @test cav_q0_custom.j == 2
        @test cav_q0_custom.x == 0.8

        marg_q0 = MarginalFP_q0(1, :zero, 0.0, rng)
        @test marg_q0.i == 1
        @test marg_q0.x == 0.0
        
        marg_q0_rand = MarginalFP_q0(1, :random, 0.0, rng)
        @test marg_q0_rand.i == 1
        @test marg_q0_rand.x == rand(rng_ref)

        marg_q0_custom = MarginalFP_q0(1, :custom, 0.8, rng)
        @test marg_q0_custom.i == 1
        @test marg_q0_custom.x == 0.8

        node = NodeFP(1, [2], :zero, 0.5, 0.3, 0.1, rng)  # node container wraps message collections
        @test node.i == 1
        @test node.neighs == [2]
        @test node.neighs_idx[2] == 1
        @test typeof(node.cavs[1]) == CavityFP{Deterministic, Int64, Float64}
        @test typeof(node.marg) == MarginalFP{Deterministic, Int64, Float64}

        node = NodeFP(1, [2], :random, 0.5, 0.3, 0.1, rng)  # node container wraps message collections
        @test node.i == 1
        @test node.neighs == [2]
        @test node.neighs_idx[2] == 1
        @test typeof(node.cavs[1]) == CavityFP{Deterministic, Int64, Float64}
        @test typeof(node.marg) == MarginalFP{Deterministic, Int64, Float64}

        node = NodeFP(1, [2], :custom, 0.5, 0.3, 0.1, rng)  # node container wraps message collections
        @test node.i == 1
        @test node.neighs == [2]
        @test node.neighs_idx[2] == 1
        @test typeof(node.cavs[1]) == CavityFP{Deterministic, Int64, Float64}
        @test typeof(node.marg) == MarginalFP{Deterministic, Int64, Float64}

        node_q0 = NodeFP_q0(1, [2], :zero, 0.7, rng)
        @test node_q0.i == 1
        @test node_q0.neighs == [2]
        @test node_q0.neighs_idx[2] == 1
        @test typeof(node_q0.cavs[1]) == CavityFP_q0{Deterministic, Int64, Float64}
        @test typeof(node_q0.marg) == MarginalFP_q0{Deterministic, Int64, Float64}

        node_q0 = NodeFP_q0(1, [2], :random, 0.7, rng)
        @test node_q0.i == 1
        @test node_q0.neighs == [2]
        @test node_q0.neighs_idx[2] == 1
        @test typeof(node_q0.cavs[1]) == CavityFP_q0{Deterministic, Int64, Float64}
        @test typeof(node_q0.marg) == MarginalFP_q0{Deterministic, Int64, Float64}

        node_q0 = NodeFP_q0(1, [2], :custom, 0.7, rng)
        @test node_q0.i == 1
        @test node_q0.neighs == [2]
        @test node_q0.neighs_idx[2] == 1
        @test typeof(node_q0.cavs[1]) == CavityFP_q0{Deterministic, Int64, Float64}
        @test typeof(node_q0.marg) == MarginalFP_q0{Deterministic, Int64, Float64}
    end

    @testset "Populations" begin
        rng = Xoshiro(BASE_SEED)
        rng_ref = copy(rng)
        
        pop_zero = PopFP(2, :zero, 0.0, 0.0, 0.0, rng)  # vectorised zero initialisation
        @test pop_zero.mu_pop == zeros(2)
        @test pop_zero.q_pop == zeros(2)
        @test pop_zero.chi_pop == zeros(2)
        
        pop_rand = PopFP(2, :random, 0.0, 0.0, 0.0, rng)
        @test pop_rand.mu_pop == rand(rng_ref, 2)
        @test pop_rand.q_pop == rand(rng_ref, 2)
        @test pop_rand.chi_pop == zeros(2)
        
        pop_custom = PopFP(2, :custom, 0.1, 0.2, 0.3, rng)
        @test pop_custom.q_pop == fill(0.2, 2)
        @test pop_custom.mu_pop == fill(0.1, 2)
        @test pop_custom.chi_pop == fill(0.3, 2)

        pop_q0_zero = PopFP_q0(2, :zero, 0.0, rng)  # q0 variant vector checks
        @test pop_q0_zero.x_pop == zeros(2)

        pop_q0_rand = PopFP_q0(2, :random, 0.0, rng)
        @test pop_q0_rand.x_pop == rand(rng_ref, 2)

        pop_q0_custom = PopFP_q0(2, :custom, 0.3, rng)
        @test pop_q0_custom.x_pop == fill(0.3, 2)

        pop_j = PopJ(2, 0.2, 0.3, 0.1, 2.0, rng)  # uses sample_couplings internally; verify distributional match
        J_expected = zeros(2)
        Jp_expected = zeros(2)
        for i in 1:2
            u = randn(rng_ref)
            v = randn(rng_ref)
            J_expected[i] = 0.2 / 2 + sqrt(0.3 / 2) * u
            Jp_expected[i] = 0.2 / 2 + sqrt(0.3 / 2) * (0.1 * u + sqrt(1 - 0.1^2) * v)
        end
        @test pop_j.J_pop ≈ J_expected
        @test pop_j.Jp_pop ≈ Jp_expected
    end

    @testset "Sampling" begin
        rng = Xoshiro(BASE_SEED)
        rng_ref = copy(rng)
        tsave = range(0.0, step=0.05, length=3)

        traj, tsave_out, conv, t_eq = run_MC(base_model, 0.05; rng=rng, tsave=collect(tsave))
        @test size(traj) == (2, 3)
        @test tsave_out == collect(tsave)
        @test conv
        @test t_eq ≈ tsave[end]

        traj_dis, tsave_dis, conv_dis, t_eq_dis = run_MC(base_model_dis, 0.05; rng=rng, tsave=collect(tsave))
        @test size(traj_dis) == (2, 3)
        @test tsave_dis == collect(tsave)
        @test conv_dis
        @test t_eq_dis ≈ tsave[end]

        traj_fc, tsave_fc, conv_fc, t_eq_fc = run_MC(base_model_dis_fc, 0.05; rng=rng, tsave=collect(tsave))
        @test size(traj_fc) == (2, 3)
        @test tsave_fc == collect(tsave)
        @test conv_fc
        @test t_eq_fc ≈ tsave[end]
    end

    @testset "Fixed-point solvers" begin
        nodes, conv, diverged = run_GECaM_FP(base_model, 5, 1e-3, 0.5; init_type=:custom, mu0=0.2, q0=0.1, chi0=0.05, rng=Xoshiro(BASE_SEED), regularization=1e-6)
        @test length(nodes) == 2
        @test conv isa Bool
        @test diverged isa Bool
        # only assert numeric conditions when the solver signals success
        if !diverged
            @test all(n -> n.marg.mu >= 0 && isfinite(n.marg.mu), nodes)
        end

        cav_pop, marg_pop, conv_dis, diverged_dis = run_GECaM_FP(base_model_dis, 5, 5, 1e-3, 0.5; init_type=:custom, mu0=0.2, q0=0.1, chi0=0.05, rng=Xoshiro(BASE_SEED), regularization=1e-6)
        @test length(cav_pop.mu_pop) == 5
        @test length(marg_pop.mu_pop) == 5
        @test conv_dis isa Bool
        @test diverged_dis isa Bool
        # population dynamics can exit early; guard finite checks accordingly
        if !diverged_dis
            @test all(isfinite, cav_pop.mu_pop)
        end

        nodes_q0, conv_q0, diverged_q0 = run_GECaM_FP_q0(base_model, 5, 1e-3, 0.5; init_type=:custom, x0=0.2, rng=Xoshiro(BASE_SEED), regularization=1e-6)
        @test length(nodes_q0) == 2
        @test conv_q0 isa Bool
        @test diverged_q0 isa Bool
        # q=0 variant shares the same divergence semantics
        if !diverged_q0
            @test all(n -> n.marg.x >= 0 && isfinite(n.marg.x), nodes_q0)
        end

        cav_pop_q0, marg_pop_q0, conv_dis_q0, diverged_dis_q0 = run_GECaM_FP_q0(base_model_dis, 5, 5, 1e-3, 0.5; init_type=:custom, x0=0.2, rng=Xoshiro(BASE_SEED), regularization=1e-6)
        @test length(cav_pop_q0.x_pop) == 5
        @test length(marg_pop_q0.x_pop) == 5
        @test conv_dis_q0 isa Bool
        @test diverged_dis_q0 isa Bool
        # only check numerical content when the iterative loop converges
        if !diverged_dis_q0
            @test all(isfinite, cav_pop_q0.x_pop)
        end
    end

    @testset "Analytic FC" begin
        deltas, mus, qs, chis, sigma2s, phis = analytic_FC(0.2, 0.1, 3, -0.5, 0.5)  # DMFT helper smoke test
        @test length(deltas) == 3
        @test length(mus) == 3 == length(qs) == length(chis) == length(sigma2s) == length(phis)
        @test all(isfinite, mus)
        @test all(phi -> 0 <= phi <= 1, phis)
    end
end

@testset "Aqua" begin
    Aqua.test_all(RandomLotkaVolterraCavity, deps_compat=(check_extras=false, check_weakdeps=false))  # static analysis coverage
end
