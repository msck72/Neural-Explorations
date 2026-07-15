import { BrowserRouter, Routes, Route } from "react-router-dom";

import Header from "./components/Header";
import ApplyFilters from "./components/ApplyFilters";
import Classification from "./components/Classification";
import VideoManipulations from "./components/VideoManipulations";

function App() {

  return (
    <BrowserRouter>
        <Header />

        <main className="pt-30">
          <Routes>
            <Route path="/" element={<ApplyFilters/>} />
            <Route path="/Classification" element={<Classification/>} />
            <Route path="/VideoManipulations" element={<VideoManipulations/>} />
          </Routes>
        </main>
    </BrowserRouter>
  )
}

export default App
