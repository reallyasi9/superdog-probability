using HTTP
using LazyJSON
using Gumbo
using AbstractTrees
using Gadfly
using Distributions
using Printf

istable(e::Any) = false
istable(e::HTMLElement{:table}) = begin
    a = attrs(e)
    (haskey(a, "class") && a["class"] == "results_table") ? true : false
end

istext(e::Any, s::AbstractString) = false
istext(e::HTMLText, s::AbstractString) = text(e) == s

istr(e::Any, s::AbstractString) = false
istr(e::HTMLElement{:tr}, s::AbstractString) = begin
    for child in PreOrderDFS(e)
        if istext(child, s)
            return true
        end
    end
    false
end

struct SystemResults
    rank::Integer
    name::AbstractString
    pct_correct::Real
    pct_spread::Real
    mae::Real
    bias::Real
    mse::Real
    games::Integer
    suw::Integer
    sul::Integer
    atsw::Integer
    atsl::Integer
end

elementtext(e::Any) = nothing
elementtext(e::HTMLText) = text(e)

textelements(e::HTMLElement, c::Channel) = begin
    for elem in PreOrderDFS(e)
        t = elementtext(elem)
        if !isnothing(t)
            put!(c, t)
        end
    end
    close(c)
end

SystemResults(e::HTMLElement{:tr}) = begin
    texts = [t for t in Channel((c) -> textelements(e, c))]
    SystemResults(
        parse(Int64, texts[1]),
        texts[2],
        parse(Float64, texts[3]),
        parse(Float64, texts[4]),
        parse(Float64, texts[5]),
        parse(Float64, texts[6]),
        parse(Float64, texts[7]),
        parse(Int64, texts[8]),
        parse(Int64, texts[9]),
        parse(Int64, texts[10]),
        parse(Int64, texts[11]),
        parse(Int64, texts[12])
    )
end

function getsystemresults(year::Int64, name::AbstractString)
    year2 = year - 2000
    r = HTTP.request("GET", "https://www.thepredictiontracker.com/ncaaresults.php?orderby=wpct%20desc&type=1&year=$year2")
    doc = parsehtml(String(r.body))
    local table::HTMLElement{:table}
    for elem in PreOrderDFS(doc.root)
        if istable(elem)
            table = elem
            break
        end
    end
    local tr::HTMLElement{:tr}
    for elem in PreOrderDFS(table)
        if !istr(elem, name) continue end
        tr = elem
    end
    SystemResults(tr)
end

struct DogGame
	id::Int64
    week::Int64
	points::Int64
	line::Float64
	probability::Float64
	outcome::Int64
    underdog::String
    overdog::String
end

function getdogs(year::Int64, dogs::Vector{String}, sr::SystemResults)
	std = sqrt(sr.mse - sr.bias^2)
	dist = Normal(-sr.bias, std)

	dgs = Vector{DogGame}()
	for (i, dog) in enumerate(dogs)
		week = div(i - 1, 3) + 1
		points = rem(i - 1, 3) + 3
		r = HTTP.request("GET", "https://api.collegefootballdata.com/lines?year=$year&week=$week&seasonType=regular&team=$(HTTP.URIs.escapeuri(dog))")
		lines = LazyJSON.value(String(r.body))
		if length(lines) != 1
			throw(ErrorException("line for game $dog (week $week) not found"))
		end
		line = lines[1]
		local spread::Float64
		for l in line["lines"]
			spread = parse(Float64, l["spread"])
			if l["provider"] == "consensus"
				break
			end
		end
		outcome = line["awayScore"] - line["homeScore"]
		underdog = line["homeTeam"]
		overdog = line["awayTeam"]
		if spread < 0
			underdog, overdog = overdog, underdog
		end
		prob = cdf(dist, spread)
		if spread > 0
			prob = 1-prob
		end
		push!(dgs, DogGame(line["id"], week, points, spread, prob, outcome, underdog, overdog))
	end
	dgs
end

function getdogresults(year::Int64, system::String)
	dogs = [
		"Utah State",
		"North Carolina",
		"Colorado State",
		"BYU",
		"San Diego State",
		"Southern Mississippi",
		"Air Force",
		"West Virginia",
		"North Texas",
		"Appalachian State",
		"South Carolina",
		"Pittsburgh",
		"Western Kentucky",
		"UCLA",
		"Mississippi State",
		"Boston College",
		"New Mexico",
		"Connecticut",
		"Temple",
		"Louisville",
		"Hawai'i",
		"Duke",
		"Bowling Green",
		"Tulsa",
		"Duke",
		"Kansas",
		"UNLV",
		"Akron",
		"Georgia Tech",
		"Virginia Tech",
		"Tennessee",
		"Louisville",
		"Wyoming",
		"Temple",
		"South Florida",
		"West Virginia",
		"Western Kentucky",
		"Charlotte",
		"UCLA",
		"Kansas State",
		"Northern Illinois",
		"Arkansas",
	]

	sr = getsystemresults(year, system)

	getdogs(year, dogs, sr)
end

correct(dg::DogGame) = sign(dg.line) != sign(dg.outcome)
probability(dg::DogGame) = dg.probability
value(dg::DogGame) = dg.points * correct(dg)

simulate(dg::DogGame) = dg.points * (rand() < dg.probability)
countulate(dg::DogGame) = rand() < dg.probability

function simulation(dgs::Vector{DogGame}, n::Int64, c::Channel)
	for i in 1:n
		put!(c, mapreduce(simulate, +, dgs))
	end
end

function countulation(dgs::Vector{DogGame}, n::Int64, c::Channel)
	for i in 1:n
		put!(c, mapreduce(countulate, +, dgs))
	end
end

getsimulations(dgs::Vector{DogGame}, n::Int64) = [x for x in Channel((c) -> simulation(dgs, n, c))]
getcountulations(dgs::Vector{DogGame}, n::Int64) = [x for x in Channel((c) -> countulation(dgs, n, c))]

function main()
	dgs = getdogresults(2019, "Line (opening)")
	simulations = getsimulations(dgs, 100000)
	countulations = getcountulations(dgs, 100000)
	val = mapreduce(value, +, dgs)
	cnt = mapreduce(correct, +, dgs)
	prob = mapreduce(probability, +, dgs)

	expected = mean(simulations)


	p1 = plot(
		layer(x=[val, expected],
			y=[1000, 1000],
			label=["Actual ($val)", "Expected ($(@sprintf("%.2f", expected)))"],
			Geom.label),
		layer(x=simulations,
			xintercept=[val, expected],
			Geom.vline(color=["orange", "white"], style=[:solid, :dash], size=[2pt,1pt]),
			Geom.histogram),
		Guide.xlabel("Total value of superdog games"),
		Guide.ylabel("Number of simulations"),
		Guide.yticks(ticks=nothing),
		style(background_color=colorant"white", default_color=colorant"purple"))

	draw(SVG("superdog-values.svg", 6inch, 4inch), p1)

	p2 = plot(
		layer(x=[cnt, prob],
			y=[1000, 1000],
			label=["Actual ($cnt)", "Expected ($(@sprintf("%.2f", prob)))"],
			Geom.label),
		layer(x=countulations,
			xintercept=[cnt, prob],
			Geom.vline(color=["orange", "white"], style=[:solid, :dash], size=[2pt,1pt]),
			Geom.histogram),
		Guide.xlabel("Number of correct superdog games"),
		Guide.ylabel("Number of simulations"),
		Guide.yticks(ticks=nothing),
		style(background_color=colorant"white", default_color=colorant"purple"))

	draw(SVG("superdog-counts.svg", 6inch, 4inch), p2)

	likelihood = findfirst(sort!(simulations) .>= val) / length(simulations)
	countlihood = findfirst(sort!(countulations) .>= cnt) / length(countulations)
	println("Expected value of Luke's $(length(dgs)) superdog games: $expected")
	println("Actual value: $val")
	println("Likelihood of an outcome < $val: $likelihood")

	println("Expected number of superdog victories in Luke's $(length(dgs)) superdog games: $prob")
	println("Actual number: $cnt")
	println("Likelihood of an outcome < $cnt: $countlihood")

end

main()
