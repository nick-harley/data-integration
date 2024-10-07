module CHAKRA

export option, none, fnd, geta, getp, pts, dom, cts

struct None end
none = None()
option{A} = Union{A,None}

# Abstract Types

abstract type Id end
abstract type Constituent end
abstract type Hierarchy end
abstract type Attribute{N,T} end
abstract type Property{N,T} end

# Registered Attributes and Properties 

__attributes__(::Val{N}) where N = error("No attribute named $N.")
__attributes__(N::Symbol) = __attributes__(Val{N}())
__properties__(::Val{N}) where N = error("No property named $N.")
__properties__(N::Symbol) = __properties__(Val{N}())

# Interface Operations

fnd(x::Id,h::Hierarchy) = none
fnd(x::Id,m::Module) = fnd(x,m.__data__)

geta(::Attribute,c::Constituent) = none
geta(N::Symbol,c) = geta(__attributes__(Val{N}()),c)

getp(::Property,c::Constituent) = none
getp(N::Symbol,c) = getp(__properties__(Val{N}()),c)

pts(c::Constituent) = Id[]

dom(h::Hierarchy) = Id[]
dom(m::Module) = dom(m.__data__)

#sequence(ids,kb) = [fnd(x,kb) for x in ids]
#cts(kb) = sequence(dom(kb),kb)

cts(kb) = Pair{Id,Constituent}[x=>fnd(x,kb) for x in dom(kb)]

sequence(ids,kb) = begin
    isempty(ids) && return Constituent[]
    c = fnd(ids[1],kb)
    c isa None && return none
    r = sequence(ids[2:end],kb)
    r isa None && return none
    return Constituent[c,r...]
end

# PROPERTIES

export DESCRIPTION, TYPE

struct DESCRIPTION <: Property{:DESCRIPTION,String} end
__properties__(::Val{:DESCRIPTION}) = DESCRIPTION()

struct TYPE <: Property{:TYPE,String} end
__properties__(::Val{:TYPE}) = TYPE()

end

using Main.CHAKRA

module DWI

using Main.CHAKRA
using LightXML, JSON

# Concrete Data

xml = parse_file("dwi.xml")
r = root(xml)
wfspec = r["WorkflowSpecification"][1]
wfdescription = content(r["Description"][1])
wfnodes = wfspec["WorkflowSpecificationNode"][2:7]
wfnodetypes = [content(n["WorkflowSpecificationNodeTypeID"][1]) for n in wfnodes]
properties = [n["WorkflowSpecificationNodeProperty"] for n in wfnodes]
names = [content(p[1]["Value"][1]["ValueString"][1]) for p in properties]
descriptions = [JSON.parse(content(p[4]["Value"][1]["ValueString"][1])) for p in properties]

abstract type Id <: CHAKRA.Id end

module Steps

using ..CHAKRA
import ..DWI: descriptions, names, wfnodes

struct Id{N} <: Main.DWI.Id end
struct Step{N} <: CHAKRA.Constituent 
    Step{N}() where N = N > length(wfnodes) ? none : new{N}()
end
struct Data <: CHAKRA.Hierarchy end

struct STEP_NAME <: CHAKRA.Property{:STEP_NAME,String} end
CHAKRA.__properties__(::Val{:STEP_NAME}) = STEP_NAME()

CHAKRA.fnd(::Id{N},::Data) where N = Step{N}()
CHAKRA.pts(::Step) = CHAKRA.Id[]
CHAKRA.getp(::TYPE,::Step) = "Assembly Step"
CHAKRA.getp(::DESCRIPTION,c::Step{N}) where N = descriptions[N]
CHAKRA.getp(::STEP_NAME,c::Step{N}) where N = names[N]
CHAKRA.dom(::Data) = [Id{N}() for N in 1:6]

__data__ = Data()

end

module Workflow

using ..CHAKRA
using ..Steps
import ..DWI: wfdescription

struct Id <: Main.DWI.Id end
struct WF <: CHAKRA.Constituent end
struct Data <: CHAKRA.Hierarchy end

CHAKRA.fnd(::Id,::Data) = WF()
CHAKRA.pts(::WF) = [Steps.Id{N}() for N in 1:6]
CHAKRA.getp(::TYPE,::WF) = "Workflow Specification"
CHAKRA.getp(::DESCRIPTION,c::WF) = wfdescription
CHAKRA.dom(::Data) = [Id()]

__data__ = Data()

end

struct Data <: CHAKRA.Hierarchy end

CHAKRA.fnd(x::ID,::Data) where {ID<:Id} = fnd(x,parentmodule(ID))
CHAKRA.dom(::Data) = [dom(Workflow)...,dom(Steps)...]

__data__ = Data()

end

using Main.DWI

dom(DWI)

wf = fnd(DWI.Workflow.Id(),DWI)

ps = pts(wf)

CHAKRA.sequence([ps...,DWI.Steps.Id{10}()],DWI)

s = fnd(ps[3],DWI)

getp(:STEP_NAME,s)

module Events

using Main.CHAKRA
using CSV
using DataFrames

df = DataFrame(CSV.File("STEPS.csv"))
idx = df[:,1]

# Abstract Types

struct Id{N} <: CHAKRA.Id end
struct Event{N} <: CHAKRA.Constituent end
struct Data <: CHAKRA.Hierarchy end

# Attribute Definitions

abstract type Attribute{N,T} <: CHAKRA.Attribute{N,T} end

struct TASK_NAME <: Attribute{:TASK_NAME,String} end
CHAKRA.__attributes__(::Val{:TASK_NAME}) = TASK_NAME()

struct ORIGIN_NAME <: Attribute{:ORIGIN_NAME,String} end
CHAKRA.__attributes__(::Val{:ORIGIN_NAME}) = ORIGIN_NAME()

struct START_TIME <: Attribute{:START_TIME,String} end
CHAKRA.__attributes__(::Val{:START_TIME}) = START_TIME()

struct STEP_DURATION <: Attribute{:STEP_DURATION,Float64} end
CHAKRA.__attributes__(::Val{:STEP_DURATION}) = STEP_DURATION()

struct STEP_INDEX <: Attribute{:STEP_INDEX,Int} end
CHAKRA.__attributes__(::Val{:STEP_INDEX}) = STEP_INDEX()

struct STEP_RANK <: Attribute{:STEP_RANK,Int} end
CHAKRA.__attributes__(::Val{:STEP_RANK}) = STEP_RANK()

struct USER_NAME <: Attribute{:USER_NAME,String} end
CHAKRA.__attributes__(::Val{:USER_NAME}) = USER_NAME()

struct EXPERIENCE_LEVEL <: Attribute{:EXPERIENCE_LEVEL,Int} end
CHAKRA.__attributes__(::Val{:EXPERIENCE_LEVEL}) = EXPERIENCE_LEVEL()

# CHAKRA Interface

CHAKRA.fnd(::Id{N},::Data) where N = Event{N}()
CHAKRA.geta(::A,::Event{n}) where {n,N,A<:Attribute{N}} = df[n,:][N]
CHAKRA.getp(::TYPE,::Event) = "Arkite Event"
CHAKRA.pts(c::Event) = CHAKRA.Id[]
CHAKRA.dom(::Data) = [Id{i}() for i in idx]

__data__ = Data()

end

using Main.Events
dom(Events)

e = fnd(Events.Id{1}(),Events)

pts(e)

getp(:TYPE,e)

geta(:START_TIME,e)

module Seqs

using Main.CHAKRA
using Main.Events

seqs = [1:4...]
seqparts(T) = [Events.Id{S}() for S in T*10-9:T*10]

struct Id{N} <: CHAKRA.Id end
struct Seq{N} <: CHAKRA.Constituent end
struct Data <: CHAKRA.Hierarchy end

struct SEQ_TYPE <: CHAKRA.Property{:SEQ_TYPE,String} end
CHAKRA.__properties__(::Val{:SEQ_TYPE}) = SEQ_TYPE()

CHAKRA.fnd(::Id{N},::Data) where N = Seq{N}()
CHAKRA.pts(::Seq{N}) where N = seqparts(N)
CHAKRA.getp(::TYPE,::Seq) = "Arkite Sequence"
CHAKRA.getp(::SEQ_TYPE,::Seq) = "Successful Assembly"
CHAKRA.dom(::Data) = [Id{N}() for N in seqs]

__data__ = Data()

end

module Chunks

using Main.CHAKRA
using Main.Events

chunks = [1:20...]
chunkparts(I) = [Events.Id{I*2-1}(),Events.Id{I*2}()]

struct Id{I} <: CHAKRA.Id end
struct Chunk{I} <: CHAKRA.Constituent end
struct Data <: CHAKRA.Hierarchy end

CHAKRA.fnd(::Id{I},::Data) where I = Chunk{I}()
CHAKRA.pts(::Chunk{I}) where I = chunkparts(I)
CHAKRA.getp(::TYPE,::Chunk) = "Arkite Chunk"
CHAKRA.dom(::Data) = [Id{N}() for N in chunks]

__data__ = Data()

end

module ChunkSeqs

using Main.CHAKRA
using Main.Chunks

chunkseqs = [1:4...]
parts(N) = [Chunks.Id{N*5-i}() for i in reverse([0:4...])] 

struct Id{N} <: CHAKRA.Id end
struct ChunkSeq{N} <: CHAKRA.Constituent end
struct Data <: CHAKRA.Hierarchy end

CHAKRA.fnd(::Id{N},::Data) where N = ChunkSeq{N}()
CHAKRA.pts(::ChunkSeq{N}) where N = parts(N)
CHAKRA.getp(::TYPE,::ChunkSeq) = "Chunk Sequence"
CHAKRA.dom(::Data) = [Id{N}() for N in chunkseqs]

__data__ = Data()

end

module SeqEx

using Main.CHAKRA
using Main.Seqs
using Main.DWI

wf = DWI.Workflow.Id()

execs = [1:4...]
ps(N) = CHAKRA.Id[Seqs.Id{N}(),wf]

struct Id{N} <: CHAKRA.Id end
struct Exec{N} <: CHAKRA.Constituent end
struct Data <: CHAKRA.Hierarchy end

CHAKRA.fnd(::Id{N},::Data) where N = Exec{N}()
CHAKRA.pts(::Exec{N}) where N = ps(N)
CHAKRA.getp(::TYPE,::Exec) = "isExecutionOf"
CHAKRA.dom(::Data) = [Id{N}() for N in execs]

__data__ = Data()

end

module ChunkEx

using Main.CHAKRA
using Main.Chunks
using Main.DWI

execs = [1:16...]
steps = [DWI.Steps.Id{N}() for N in 1:6]
arcs = Dict([0=>6,1=>1,2=>4,3=>5])
ps(N) = CHAKRA.Id[Chunks.Id{N}(),steps[arcs[N%4]]]

struct Id{N} <: CHAKRA.Id end
struct Exec{N} <: CHAKRA.Constituent end
struct Data <: CHAKRA.Hierarchy end

CHAKRA.fnd(::Id{N},::Data) where N = Exec{N}()
CHAKRA.pts(::Exec{N}) where N = ps(N)
CHAKRA.getp(::TYPE,::Exec) = "isExecutionOf"
CHAKRA.dom(::Data) = [Id{N}() for N in execs]

__data__ = Data()

end

module KB

using Main.CHAKRA
using Main.Events
using Main.Seqs
using Main.Chunks
using Main.ChunkSeqs
using Main.DWI
using Main.SeqEx
using Main.ChunkEx

struct Data <: CHAKRA.Hierarchy end

CHAKRA.fnd(x::ID,::Data) where {ID<:CHAKRA.Id} = fnd(x,parentmodule(ID))
CHAKRA.dom(::Data) = [dom(ChunkEx)...,dom(SeqEx)...,dom(ChunkSeqs)...,dom(Chunks)...,dom(Seqs)...,dom(Events)...,dom(DWI)...]

__data__ = Data()

end

using .KB
dom(KB)
fnd(Main.ChunkSeqs.Id{1}(),KB)

PAIR(X,Y) = Pair{x,y} where {x<:X,y<:Y}
LST(X) = Vector{x} where {x<:X}
OPT(X) = Union{X,CHAKRA.None}
ID = CHAKRA.Id
C = CHAKRA.Constituent
CON = PAIR(ID,C)
G = LST(CON)

domain(cs::G) = first.(cs)

mem(x::ID,cs::G)::Bool = x in domain(cs)
mem(xs::LST(ID),cs::G) = all([mem(x,cs) for x in xs])

position(x::ID,cs::G) = findfirst(==(x),domain(cs))

lookup(x::ID,cs::G) = mem(x,cs) ? cs[position(x,cs)] : none

lookup(xs::LST(ID),cs::G) = begin
    res = Pair{CHAKRA.Id,CHAKRA.Constituent}[]
    for x in xs
        c = lookup(x,cs)
        c == none && return none
        union!(res,[c])
    end
    return res
end

pop(x::ID,cs::G) = begin 
    !mem(x,cs) && return CON[]
    return cs[position(x,cs):end]
end

parts(x::ID,cs::G) = begin 
    !mem(x,cs) && return none
    c = lookup(x,cs)
    return lookup(pts(c[2]),cs)
end
parts(c::CON,cs::G) = lookup(pts(c[2]),cs)

subparts(x::ID,cs::G) = begin
    isempty(cs) && return CON[]
    c = lookup(x,cs)
    c == none && return CON[]
    res = CON[c]
    ps = pts(c[2])
    for p in ps
        union!(res,subparts(p,pop(p,cs)))
    end
    return res
end

subparts(xs::LST(ID),cs::G) = begin
    res = CON[]
    [union!(res,subparts(x,cs)) for x in xs]
    return [c for c in cs if c in res]
end

superparts(x::ID,cs::G) = begin
    isempty(cs) && return CON[]
    c = lookup(x,cs)
    c == none && return CON[]
    res = CON[c]
    for c2 in reverse(cs[1:position(x,cs)])
        if x in pts(c2[2])
            push!(res,c2)
            union!(res,superparts(c2[1],cs))
        end
    end
    return reverse(res)
end

superparts(xs::LST(ID),cs::G) = begin 
    res = CON[]
    [union!(res,superparts(x,cs)) for x in xs]
    return [c for c in cs if c in res]
end

haspart(x::ID,y::ID,cs::G) = begin
    !(mem(x,cs)) && return false
    return y in first.(parts(x,cs))
end

ispartof(x::ID,y::ID,cs::G) = haspart(y,x,cs)

hassubpart(x,y,cs) = begin
    y in first.(subparts(x,cs))
end

issubpartof(x::ID,y::ID,cs::G) = hassubpart(y,x,cs)

isdag(cs::G) = begin 
    isempty(cs) && return true
    x,c = cs[1]
    t = cs[2:end]
    ps = pts(c)
    mem(x,t) && (print("!!!"); return false)
    !mem(ps,t) && (print("!!!$x"); return false)
    !isdag(t) && (print("!!!"); return false)
    return true
end
isleaf(x::ID,cs::G) = begin 
    c = lookup(x,cs) 
    return c != none && isempty(pts(c[2]))
end
istree(cs::G) = begin
    isempty(cs) && return true
    x,c = cs[1]
    t = cs[2:end]
    sps = [subparts(p[1],cs) for p in parts(x,cs)]
    for p in parts(x)
        return isleaf(p,cs) || istree(p,cs)
    end
end

wf = DWI.Workflow.Id()::ID
xs = dom(KB);
cs = cts(KB);
x = xs[1]

cs

position(xs[10],cs)

mem(x,cs)

lookup(x,cs)

lookup(xs[1:20],cs)

pop(xs[50],cs)

xs[1]
pts(lookup(xs[1],cs)[2])

parts(xs[1],cs)

subparts(x,cs)

subparts(xs[1:2],cs)

superparts(DWI.Steps.Id{6}(),cs)

mem(SeqEx.Id{1}(),cs)

haspart(ChunkSeqs.Id{1}(),Chunks.Id{1}(),cs)

haspart(DWI.Workflow.Id(),DWI.Steps.Id{1}(),cs)

haspart(SeqEx.Id{1}(),Events.Id{5}(),cs)

hassubpart(SeqEx.Id{1}(),Events.Id{5}(),cs)

hassubpart(SeqEx.Id{1}(),Events.Id{15}(),cs)

isdag(cs)

PTY{N,T} = CHAKRA.Property{N,t} where {t<:T}
ATT{N,T} = CHAKRA.Attribute{N,t} where {t<:T}

# READ(A) : G -> option A x G 

abstract type READ{A} end

(R::READ{A})(cs::G) where A = R.op(cs)::option{A}

struct Read{A} <: READ{A}
    op::Function
    Read(A,f) = new{A}(f)
    Read(A) = new{A}(cs->none)
end

abstract type KLEISLI{A,B} end

function (K::KLEISLI{A,B})(a) where {A,B}
    a==none && return Read(B)
    Read(B,K.op(a))
end

function(K::KLEISLI{A,B})(a::A,cs::G) where {A,B}
    K(a)(cs)
end

struct BIND{A,B} <: READ{B}
    op
    BIND(R::READ{A},F::KLEISLI{A,B}) where {A,B} = new{A,B}(cs->F(R(cs),cs))
end

struct FISH{A,B} <: KLEISLI{A,B}
    op
    FISH(F::KLEISLI{A,B},G::KLEISLI{B,C}) where {A,B,C} = new{A,C}(x->cs->G(F(x,cs),cs))
end

FISH(f::KLEISLI,gs::KLEISLI...) = isempty(gs) ? f : FISH(f,FISH(gs...))

QUERY(R::READ,KS::KLEISLI...) = BIND(R,FISH(KS...))

struct PTS <: KLEISLI{CON,LST(CON)}
    op
    PTS() = new(c->cs->parts(c,cs))
end

struct GETA{N,T} <: KLEISLI{CON,T}
    op
    GETA(a::ATT{N,T}) where {N,T} = new{N,T}(c->cs->geta(a,c[2]))
    GETA(N::Symbol) = GETA(CHAKRA.__attributes__(N))
end

struct GETP{N,T} <: KLEISLI{CON,T}
    op
    GETP(a::PTY{N,T}) where {N,T} = new{N,T}(c->cs->getp(a,c[2]))
    GETP(N::Symbol) = GETP(CHAKRA.__properties__(N))
end

struct HASA <: KLEISLI{CON,Bool}
    op
    HASA(N::Symbol,v) = new(c->cs->geta(N,c[2])==v)
    HASA(N::Symbol) = new(c->cs->geta(N,c[2])!=none)
end

struct HASP <: KLEISLI{CON,Bool}
    op
    HASP(N::Symbol,v) = new(c->cs->getp(N,c[2])==v)
    HASP(N::Symbol) = new(c->cs->getp(N,c[2])!=none)
end

struct FIRSTPART <: KLEISLI{CON,Bool}
    op
    FIRSTPART(T::KLEISLI) = new(c->cs->(ps = parts(c,cs); isempty(ps) ? false : T(first(ps))(cs)))
    FIRSTPART(x::ID) = FIRSTPART(c->cs->c[1]==x)
end

struct LASTPART <: KLEISLI{CON,Bool}
    op
    LASTPART(T) = new(c->cs->(ps = parts(c,cs); isempty(ps) ? false : T(last(ps))(cs)))
    LASTPART(x::ID) = LASTPART(c->cs->c[1]==x)
end

struct ALLPARTS <: KLEISLI{CON,Bool}
    op
    ALLPARTS(T) = new(c->cs->all([T(p,cs) for p in parts(c,cs)])) 
end

struct SOMEPART <: KLEISLI{CON,Bool}
    op
    SOMEPART(T) = new(c->cs->any([T(p,cs) for p in parts(c,cs)]))
end

struct AND <: KLEISLI{CON,Bool}
    op
    AND(f::KLEISLI,g::KLEISLI) = new(a->cs->f(a,cs)&&g(a,cs))
    AND(f::KLEISLI,gs::KLEISLI...) = isempty(gs) ? f : AND(AND(f,gs[1]),gs[2:end]...)
end

struct SELECT <: READ{G}
    op
    SELECT(T::KLEISLI{CON,Bool}) = new(cs->[c for c in cs if T(c)(cs)])
end

opmap(B,f,l) = begin
    isempty(l) && return B[]
    h = f(l[1])
    t = opmap(B,f,l[2:end])
    h == none && return none
    t == none && return none
    return B[h,t...]
end

struct MAP{A,B} <: KLEISLI{LST(A),LST(B)}
    op
    MAP(f::KLEISLI{A,B}) where {A,B} = new{A,B}(l->cs->opmap(B,x->f(x)(cs),l))
end

struct RET{A} <: READ{A}
    op
    RET(v::A) where A = new{A}(_->v)
end

struct SUBCS <: KLEISLI{CON,G}
    op
    SUBCS() = new(c->cs->subparts(c,cs))
end

struct PROJ <: KLEISLI{CON,G}
    op
    PROJ(T::KLEISLI{CON,Bool}) = new(c->cs->CON[x for x in subparts(c[1],cs) if T(x,cs)])
end

struct SUM <: KLEISLI{LST(Float64),Float64}
    op
    SUM() = new(l->cs->sum(l))
end

struct AVG <: KLEISLI{LST(Float64),Float64}
    op
    AVG() = new(l->cs->sum(l)/length(l))
end

P = HASP(:TYPE,"isExecutionOf")
@time cs |> SELECT(P)

P = AND(HASP(:TYPE,"Arkite Sequence"),
        HASP(:SEQ_TYPE,"Successful Assembly"))
@time cs |> SELECT(FIRSTPART(P))

P = LASTPART(wf)
@time cs |> SELECT(P)

S = SELECT(AND(HASP(:TYPE,"isExecutionOf"),
               FIRSTPART(AND(HASP(:TYPE,"Arkite Sequence"),
                             HASP(:SEQ_TYPE,"Successful Assembly"))),
               LASTPART(wf)))
@time es = cs |> S

@time s = cs |> PROJ(HASP(:TYPE,"Arkite Event"))(es[1])

@time ds = cs |> MAP(GETA(:STEP_DURATION))(s)

@time cs |> SUM()(ds)

@time cs |> QUERY(SELECT(AND(HASP(:TYPE,"isExecutionOf"),
                       FIRSTPART(AND(HASP(:TYPE,"Arkite Sequence"),
                                     HASP(:SEQ_TYPE,"Successful Assembly"))),
                       LASTPART(DWI.Workflow.Id()))),
            MAP(FISH(PROJ(HASP(:TYPE,"Arkite Event")),MAP(GETA(:STEP_DURATION)),SUM())),
            AVG())

@time cs |> QUERY(SELECT(AND(HASP(:TYPE,"isExecutionOf"),
                             FIRSTPART(HASP(:TYPE,"Arkite Chunk")),
                             LASTPART(DWI.Steps.Id{6}()))),
                  MAP(FISH(PROJ(HASP(:TYPE,"Arkite Event")),MAP(GETA(:STEP_DURATION)),SUM())),AVG())

function chparse(x) 
    if x isa Dict
        op = x["op"]
        args = x["args"]
        return eval(Symbol(op))([chparse(a) for a in args]...)
    end
    return eval(Meta.parse(x))
end;

import JSON
d = JSON.parsefile("query.json");

x = chparse(d);

cs |> x

function EXECUTE(d::Dict,kb)
    cts(kb) |> chparse(d)
end
function EXECUTE(q::String,kb)
    d = JSON.parsefile(q)
    return EXECUTE(d,kb)
end

EXECUTE(d,KB)
