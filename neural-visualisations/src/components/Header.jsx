function Header() {
    return (
        <header className="text-center m-10 flex justify-between">
            <div className="tracking-tight">
                <h1 className="text-3xl font-bold">My App</h1>
            </div>

            <nav>
                <ul className="flex space-x-4">
                    <li><a href="#" className="text-gray-700 hover:text-gray-900">Home</a></li>
                    <li><a href="#" className="text-gray-700 hover:text-gray-900">About</a></li>
                    <li><a href="#" className="text-gray-700 hover:text-gray-900">Contact</a></li>
                </ul>
            </nav>

        </header>
    )
}

export default Header