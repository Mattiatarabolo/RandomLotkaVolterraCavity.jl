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

        cav_zero = CavityFP(1, 2, :zero, 0.0, 0.0, 0.0, Xoshiro(BASE_SEED))  # zero initialisation
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

        marg_zero = MarginalFP(1, :zero, 0.0, 0.0, 0.0, Xoshiro(BASE_SEED))
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

        cav_BP = CavityFP_BP(1, 2, :zero, 0.0, 0.0, Xoshiro(BASE_SEED))
        @test cav_BP.i == 1
        @test cav_BP.j == 2
        @test cav_BP.mu == 0.0
        @test cav_BP.q == 0.0

        cav_BP_rand = CavityFP_BP(1, 2, :random, 0.0, 0.0, rng)
        @test cav_BP_rand.i == 1
        @test cav_BP_rand.j == 2
        @test cav_BP_rand.mu == rand(rng_ref)
        @test cav_BP_rand.q == rand(rng_ref)

        cav_BP_custom = CavityFP_BP(1, 2, :custom, 0.8, 0.2, Xoshiro(BASE_SEED))
        @test cav_BP_custom.i == 1
        @test cav_BP_custom.j == 2
        @test cav_BP_custom.mu == 0.8
        @test cav_BP_custom.q == 0.2

        marg_BP = MarginalFP_BP(1, :zero, 0.0, 0.0, Xoshiro(BASE_SEED))
        @test marg_BP.i == 1
        @test marg_BP.mu == 0.0
        @test marg_BP.q == 0.0
        
        marg_BP_rand = MarginalFP_BP(1, :random, 0.0, 0.0, rng)
        @test marg_BP_rand.i == 1
        @test marg_BP_rand.mu == rand(rng_ref)
        @test marg_BP_rand.q == rand(rng_ref)

        marg_BP_custom = MarginalFP_BP(1, :custom, 0.8, 0.2, Xoshiro(BASE_SEED))
        @test marg_BP_custom.i == 1
        @test marg_BP_custom.mu == 0.8
        @test marg_BP_custom.q == 0.2

        node = NodeFP(1, [2], :zero, 0.5, 0.3, 0.1, Xoshiro(BASE_SEED))  # node container wraps message collections
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

        node = NodeFP(1, [2], :custom, 0.5, 0.3, 0.1, Xoshiro(BASE_SEED))  # node container wraps message collections
        @test node.i == 1
        @test node.neighs == [2]
        @test node.neighs_idx[2] == 1
        @test typeof(node.cavs[1]) == CavityFP{Deterministic, Int64, Float64}
        @test typeof(node.marg) == MarginalFP{Deterministic, Int64, Float64}

        node_IBMF = NodeFP_IBMF(1, [2], :zero, 0.7, Xoshiro(BASE_SEED))
        @test node_IBMF.i == 1
        @test node_IBMF.neighs == [2]

        node_IBMF = NodeFP_IBMF(1, [2], :random, 0.7, rng)
        @test node_IBMF.i == 1
        @test node_IBMF.neighs == [2]

        node_IBMF = NodeFP_IBMF(1, [2], :custom, 0.7, Xoshiro(BASE_SEED))
        @test node_IBMF.i == 1
        @test node_IBMF.neighs == [2]

        node_BP = NodeFP_BP(1, [2], :zero, 0.5, 0.3, Xoshiro(BASE_SEED))  # node container wraps message collections
        @test node_BP.i == 1
        @test node_BP.neighs == [2]
        @test node_BP.neighs_idx[2] == 1
        @test typeof(node_BP.cavs[1]) == CavityFP_BP{Deterministic, Int64, Float64}
        @test typeof(node_BP.marg) == MarginalFP_BP{Deterministic, Int64, Float64}

        node_BP_rand = NodeFP_BP(1, [2], :random, 0.5, 0.3, rng)  # node container wraps message collections
        @test node_BP_rand.i == 1
        @test node_BP_rand.neighs == [2]
        @test node_BP_rand.neighs_idx[2] == 1
        @test typeof(node_BP_rand.cavs[1]) == CavityFP_BP{Deterministic, Int64, Float64}
        @test typeof(node_BP_rand.marg) == MarginalFP_BP{Deterministic, Int64, Float64}

        node_BP_custom = NodeFP_BP(1, [2], :custom, 0.5, 0.3, Xoshiro(BASE_SEED))  # node container wraps message collections
        @test node_BP_custom.i == 1
        @test node_BP_custom.neighs == [2]
        @test node_BP_custom.neighs_idx[2] == 1
        @test typeof(node_BP_custom.cavs[1]) == CavityFP_BP{Deterministic, Int64, Float64}
        @test typeof(node_BP_custom.marg) == MarginalFP_BP{Deterministic, Int64, Float64}
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

        pop_IBMF_zero = RandomLotkaVolterraCavity.init_pop_IBMF(2, :zero, 0.0, rng)  # q0 variant vector checks
        @test pop_IBMF_zero == zeros(2)

        pop_IBMF_rand = RandomLotkaVolterraCavity.init_pop_IBMF(2, :random, 0.0, rng)
        @test pop_IBMF_rand == rand(rng_ref, 2)
        pop_IBMF_custom = RandomLotkaVolterraCavity.init_pop_IBMF(2, :custom, 0.3, rng)
        @test pop_IBMF_custom == fill(0.3, 2)

        pop_BP_zero = PopFP_BP(2, :zero, 0.0, 0.0, rng)  # vectorised zero initialisation
        @test pop_BP_zero.mu_pop == zeros(2)
        @test pop_BP_zero.q_pop == zeros(2)
        
        pop_BP_rand = PopFP_BP(2, :random, 0.0, 0.0, rng)
        @test pop_BP_rand.mu_pop == rand(rng_ref, 2)
        @test pop_BP_rand.q_pop == rand(rng_ref, 2)
        
        pop_BP_custom = PopFP_BP(2, :custom, 0.1, 0.2, rng)
        @test pop_BP_custom.q_pop == fill(0.2, 2)
        @test pop_BP_custom.mu_pop == fill(0.1, 2)
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

        nodes_IBMF, conv_IBMF, diverged_IBMF = run_IBMF_FP(base_model, 5, 1e-3, 0.5; init_type=:custom, x0=0.2, rng=Xoshiro(BASE_SEED), regularization=1e-6)
        @test length(nodes_IBMF) == 2
        @test conv_IBMF isa Bool
        @test diverged_IBMF isa Bool
        # q=0 variant shares the same divergence semantics
        if !diverged_IBMF
            @test all(n -> n.x >= 0 && isfinite(n.x), nodes_IBMF)
        end

        pop_IBMF, conv_dis_IBMF, diverged_dis_IBMF = run_IBMF_FP(base_model_dis, 5, 5, 1e-3, 0.5; init_type=:custom, x0=0.2, rng=Xoshiro(BASE_SEED), regularization=1e-6)
        @test length(pop_IBMF) == 5
        @test conv_dis_IBMF isa Bool
        @test diverged_dis_IBMF isa Bool
        # only check numerical content when the iterative loop converges
        if !diverged_dis_IBMF
            @test all(isfinite, pop_IBMF)
        end

        nodes_BP, conv_BP, diverged_BP = run_BP_FP(base_model, 5, 1e-3, 0.5; init_type=:custom, mu0=0.2, q0=0.1, rng=Xoshiro(BASE_SEED), regularization=1e-6)
        @test length(nodes_BP) == 2
        @test conv_BP isa Bool
        @test diverged_BP isa Bool
        # only assert numeric conditions when the solver signals success
        if !diverged_BP
            @test all(n -> n.marg.mu >= 0 && isfinite(n.marg.mu), nodes_BP)
        end

        cav_pop_BP, marg_pop_BP, conv_dis_BP, diverged_dis_BP = run_BP_FP(base_model_dis, 5, 5, 1e-3, 0.5; init_type=:custom, mu0=0.2, q0=0.1, rng=Xoshiro(BASE_SEED), regularization=1e-6)
        @test length(cav_pop_BP.mu_pop) == 5
        @test length(marg_pop_BP.mu_pop) == 5
        @test conv_dis_BP isa Bool
        @test diverged_dis_BP isa Bool
        # population dynamics can exit early; guard finite checks accordingly
        if !diverged_dis_BP
            @test all(isfinite, cav_pop_BP.mu_pop)
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
