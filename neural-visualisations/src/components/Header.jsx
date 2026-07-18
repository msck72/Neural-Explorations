import { Link } from "react-router-dom";

function Header() {
    return (
        <header className="fixed inset-x-0 top-5 z-50 mx-auto flex max-w-7xl items-center justify-between rounded-full border border-white/10 bg-slate-950/95 px-6 py-4 text-slate-100 shadow-xl shadow-slate-950/30 backdrop-blur-xl">
            <div className="tracking-tight">
                <h1 className="text-3xl font-bold text-white">Inference Engine</h1>
            </div>

            <nav>
                <ul className="flex items-center gap-4 text-sm font-medium text-slate-300">
                    <li><Link className="transition hover:text-cyan-300" to="/">Apply Filters</Link></li>
                    <li><Link className="transition hover:text-cyan-300" to="/Classification">Classification</Link></li>
                    <li><Link className="transition hover:text-cyan-300" to="/VideoManipulations">Video Manipulations</Link></li>
                </ul>
            </nav>

        </header>
    )
}

export default Header