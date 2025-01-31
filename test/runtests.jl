using RandomLotkaVolterraCavity
using Test
using Aqua

@testset "Aqua" begin
    Aqua.test_all(RandomLotkaVolterraCavity, deps_compat=(check_extras=false, check_weakdeps=false))    
end
