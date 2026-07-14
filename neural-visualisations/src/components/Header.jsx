import { Link } from "react-router-dom";

function Header() {
    return (
        <header className="text-center m-10 flex justify-between">
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