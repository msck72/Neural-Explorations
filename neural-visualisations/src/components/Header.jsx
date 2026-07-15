import { Link } from "react-router-dom";

function Header() {
    return (
        <header className="fixed px-10 top-5 py-5 pb-10 h-16 w-full text-center flex justify-between bg-white shadow z-50">
            <div className="tracking-tight">
                <h1 className="text-3xl font-bold">Inference Engine</h1>
            </div>

            <nav>
                <ul className="flex space-x-4">
                    <li><Link to="/">Apply Filters</Link></li>
                    <li><Link to="/Classification">Classification</Link></li>
                    <li><Link to="/VideoManipulations">Video Manipulations</Link></li>
                </ul>
            </nav>

        </header>
    )
}

export default Header