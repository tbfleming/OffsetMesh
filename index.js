// Todd Fleming 2016
//
// Offsets a triangle mesh (STL format) using the GPU. Output is a height map.

'use strict';

let offsetVertexShader = `
    precision highp float;
    precision highp int;

    uniform mat4 projectionMatrix;
    uniform mat4 modelViewMatrix;
    uniform float offset;

    attribute vec3 position1;
    attribute vec3 position2;
    attribute vec3 position3;
    attribute float vertexIndex;

    varying float vIsTriangle;
    varying vec3 vPosition;
    varying vec3 vPosition1;
    varying vec3 vPosition2;
    varying vec3 vPosition3;

    vec3 project(vec3 p) {
        return (projectionMatrix * modelViewMatrix * vec4(p, 1.0)).xyz;
    }

    void main() {
        vec3 p1 = project(position1);
        vec3 p2 = project(position2);
        vec3 p3 = project(position3);

        vec4 result;

        int index = int(vertexIndex);

        if(index < 6) {
            vec2 minBounds = min(min(
                vec2(p1.x - offset, p1.y - offset),
                vec2(p2.x - offset, p2.y - offset)),
                vec2(p3.x - offset, p3.y - offset));
            vec2 maxBounds = max(max(
                vec2(p1.x + offset, p1.y + offset),
                vec2(p2.x + offset, p2.y + offset)),
                vec2(p3.x + offset, p3.y + offset));

            if(index == 0)
                result = vec4(minBounds.x, minBounds.y, p1.z, 1.0);
            else if(index == 1 || index == 4)
                result = vec4(maxBounds.x, minBounds.y, p1.z, 1.0);
            else if(index == 2 || index == 3)
                result = vec4(minBounds.x, maxBounds.y, p1.z, 1.0);
            else
                result = vec4(maxBounds.x, maxBounds.y, p1.z, 1.0);
        } else {
            vec3 triangleOffset = offset * normalize(cross(p2 - p1, p3 - p1));
            if(index == 7)
                result = vec4(p1 + triangleOffset, 1.0);
            else if(index == 8)
                result = vec4(p2 + triangleOffset, 1.0);
            else
                result = vec4(p3 + triangleOffset, 1.0);
        }

        gl_Position = result;
        vIsTriangle = float(index >= 6);
        vPosition = result.xyz;
        vPosition1 = p1;
        vPosition2 = p2;
        vPosition3 = p3;
    }
`;

let offsetFragmentShader = `
    #extension GL_EXT_frag_depth: enable

    precision highp float;
    precision highp int;

    uniform float offset;

    varying float vIsTriangle;
    varying vec3 vPosition;
    varying vec3 vPosition1;
    varying vec3 vPosition2;
    varying vec3 vPosition3;

    vec3 debugNormal;

    bool found = false;
    float foundZ = -100.0;

    void sphere(vec3 p) {
        float r = offset;
        vec2 delta = vPosition.xy - p.xy;
        if(length(delta) > r)
            return;

        float deltaZ = sqrt(r * r - delta.x * delta.x - delta.y * delta.y);
        float z = p.z + deltaZ;
        if(z < foundZ)
            return;
        foundZ = z;
        found = true;
        debugNormal = normalize(vec3(delta.xy, deltaZ));
    }

    void cyl(vec3 p1, vec3 p2) {
        float r = offset;
        if(p1.xy == p2.xy)
            return;
        vec3 B = normalize(p2 - p1);
        vec3 C = vPosition - p1;
        float a = dot(B.xy, B.xy);
        float b = -2.0 * B.z * dot(B.xy, C.xy);
        float w = C.x * B.y - C.y * B.x;
        float c = C.y * C.y * B.z * B.z + B.z * B.z * C.x * C.x + w * w - r * r;
        float sq = b * b - 4.0 * a * c;
        if(sq < 0.0)
            return;
        C.z = (-b + sqrt(sq)) / 2.0 / a;

        float l = dot(C, B);
        if(l < 0.0 || l > distance(p1, p2))
            return;

        float z = p1.z + C.z;
        if(z < foundZ)
            return;
        foundZ = z;
        found = true;
        debugNormal = normalize(vec3(1.0, 1.0, 1.0));
    }

    void main() {
        vec3 p1 = vPosition1;
        vec3 p2 = vPosition2;
        vec3 p3 = vPosition3;

        if(vIsTriangle == 0.0) {
            sphere(p1);
            sphere(p2);
            sphere(p3);
            cyl(p1, p2);
            cyl(p1, p3);
            cyl(p2, p3);
        } else {
            foundZ = vPosition.z;
            found = true;
            debugNormal = normalize(cross(p2 -p1, p3 -p1));
        }

        if(found) {
            foundZ = clamp(foundZ, -1.0, 1.0);
            gl_FragDepthEXT = -foundZ / 2.0 + 0.5;
            float z = floor((foundZ + 1.0) * float(0xffff) / 2.0 + 0.5);
            int high = int(floor(z / 256.0));
            int low = int(z) - (high * 256);
            gl_FragColor = vec4(float(high) / 255.0, float(low) / 255.0, 0.0, 1.0);

            //float debugLight = dot(debugNormal, normalize(vec3(1.0, 1.0, 1.0))) / 2.0 +0.5;
            //gl_FragColor = vec4(vec3(0.0, 1.0, 1.0) * debugLight, 1.0);
        } else
            discard;
    }
`;

let offsetRenderer = new THREE.WebGLRenderer();

function createOffsetHeightMap(vertexes, offset, resolution) {
    let startTime = new Date().getTime();

    let position = new Float32Array(vertexes.length * 3);
    let position1 = new Float32Array(vertexes.length * 3);
    let position2 = new Float32Array(vertexes.length * 3);
    let position3 = new Float32Array(vertexes.length * 3);
    let vertexIndex = new Int8Array(vertexes.length);
    for (let i = 0; i < vertexes.length; i += 9) {
        for (let j = 0; j < 27; ++j) {
            position[3 * i + j] = vertexes[i + j % 9];
            position1[3 * i + j] = vertexes[i + j % 3 + 0];
            position2[3 * i + j] = vertexes[i + j % 3 + 3];
            position3[3 * i + j] = vertexes[i + j % 3 + 6];
        }
        for (let j = 0; j < 9; ++j)
            vertexIndex[i + j] = j;
    }

    let geometry = new THREE.BufferGeometry();
    geometry.addAttribute('position', new THREE.BufferAttribute(position, 3));
    geometry.addAttribute('position1', new THREE.BufferAttribute(position1, 3));
    geometry.addAttribute('position2', new THREE.BufferAttribute(position2, 3));
    geometry.addAttribute('position3', new THREE.BufferAttribute(position3, 3));
    geometry.addAttribute('vertexIndex', new THREE.BufferAttribute(vertexIndex, 1));

    let box = new THREE.Box3();
    box.setFromArray(vertexes);
    let maxSize = Math.max(box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z);
    let scale = 2 / (maxSize + 2 * offset);

    let offsetMaterial;
    offsetMaterial = new THREE.RawShaderMaterial({
        uniforms: {
            "offset": { value: offset * scale },
        },
        vertexShader: offsetVertexShader,
        fragmentShader: offsetFragmentShader,
    });
    offsetMaterial.extensions.fragDepth = true;
    let object = new THREE.Mesh(geometry, offsetMaterial);

    let center = (new THREE.Vector3()).addVectors(box.min, box.max).multiplyScalar(scale / 2);
    let camera = new THREE.Camera();
    let e = camera.projectionMatrix.elements;
    e[0] = scale; e[4] = 0; e[8] = 0; e[12] = -center.x;
    e[1] = 0; e[5] = scale; e[9] = 0; e[13] = -center.y;
    e[2] = 0; e[6] = 0; e[10] = scale; e[14] = -center.z;
    e[3] = 0; e[7] = 0; e[11] = 0; e[15] = 1;

    let offsetScene = new THREE.Scene();
    offsetScene.add(object);
    let rtTexture = new THREE.WebGLRenderTarget(resolution, resolution, {});

    offsetRenderer.render(offsetScene, camera, rtTexture, true);

    let rawHeightMap = new Uint8Array(resolution * resolution * 4);
    offsetRenderer.readRenderTargetPixels(rtTexture, 0, 0, resolution, resolution, rawHeightMap);

    let heightMap = new Float32Array(resolution * resolution);
    for (let x = 0; x < resolution; ++x) {
        for (let y = 0; y < resolution; ++y) {
            let r = rawHeightMap[x * 4 + y * resolution * 4];
            let g = rawHeightMap[x * 4 + y * resolution * 4 + 1];
            heightMap[x + y * resolution] = ((r << 8) + g) / 0xffff * 2.0 - 1.0;
        }
    }

    let endTime = new Date().getTime();
    document.getElementById('speed1').textContent =
        vertexes.length / 3 + ' triangles in ' + (endTime - startTime) + ' ms';

    return {
        scale: scale,
        center: center,
        rawHeightMap: rawHeightMap,
        heightMap: heightMap,
    };
} // createOffsetHeightMap

let renderer, scene;
function initDisplay() {
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setClearColor(0xf0f0f0);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.sortObjects = false;

    let container = document.createElement('div');
    document.body.appendChild(container);
    container.appendChild(renderer.domElement);
    window.addEventListener('resize', onWindowResize, false);

    let camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 1, 10000);
    camera.position.x = 0;
    camera.position.y = -200;
    camera.position.z = 100;

    let controls = new THREE.TrackballControls(camera, container);
    controls.rotateSpeed = 10.0;

    scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0x505050));

    let light = new THREE.SpotLight(0xffffff, 1.5);
    light.position.set(0, 500, 2000);
    scene.add(light);

    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    function render() {
        requestAnimationFrame(render);
        controls.update();
        renderer.render(scene, camera);
    }

    render();
} // initDisplay

initDisplay();

let offset = 10;
let geometry, mesh;
let lines = [];

function setGemotry(g) {
    if (mesh)
        scene.remove(mesh);
    let material = new THREE.MeshLambertMaterial({ color: 0xffff00 });
    mesh = new THREE.Mesh(g, material);
    scene.add(mesh);
    processGeometry(g)
}

function processGeometry(g) {
    let startTotalTime = new Date().getTime();

    if (!(g instanceof THREE.BufferGeometry))
        g = (new THREE.BufferGeometry()).fromGeometry(g);
    geometry = g;

    let position = g.getAttribute('position').array;
    let resolution = 1024;
    let hm = createOffsetHeightMap(position, offset, resolution);

    for(let line of lines) {
        scene.remove(line);
    }
    lines = [];

    //let texture = new THREE.DataTexture(hm.rawHeightMap, resolution, resolution, THREE.RGBAFormat, THREE.UnsignedByteType);
    //texture.needsUpdate = true;
    //let textureMesh = new THREE.Mesh(new THREE.PlaneBufferGeometry(100, 100, 1, 1), new THREE.MeshBasicMaterial({ map: texture }));
    //scene.add(textureMesh);

    let startTime = new Date().getTime();

    for (let j = 0; j < resolution; j += Math.floor(resolution / 150)) {
        let y = ((j * 2 / (resolution - 1) - 1) + hm.center.y) / hm.scale;
        let position = new Float32Array(resolution * 3);
        for (let i = 0; i < resolution; ++i) {
            let x = ((i * 2 / (resolution - 1) - 1) + hm.center.x) / hm.scale;
            let z = (hm.heightMap[i + j * resolution] + hm.center.z) / hm.scale;
            position[i * 3 + 0] = x;
            position[i * 3 + 1] = y;
            position[i * 3 + 2] = z;
        }
        let g = new THREE.BufferGeometry();
        g.addAttribute('position', new THREE.BufferAttribute(position, 3));
        let line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x0000ff }));
        scene.add(line);
        lines.push(line);
    }

    let endTime = new Date().getTime();
    document.getElementById('speed2').textContent =
        lines.length + ' lines in ' + (endTime - startTime) + ' ms';

    let endTotalTime = new Date().getTime();
    document.getElementById('speed3').textContent =
        'Total time: ' + (endTotalTime - startTotalTime) + ' ms';
} // processGeometry

let offsetElement = document.getElementById('offset');
offsetElement.value = offset;
offsetElement.onchange = e => {
    offset = Math.max(0, e.target.value);
    e.target.value = offset;
    if (geometry)
        processGeometry(geometry);
};

document.getElementById('file').onchange = e => {
    let reader = new FileReader;
    reader.onload = () => {
        let g = (new THREE.STLLoader()).parse(reader.result);
        setGemotry(g);
    };
    reader.readAsArrayBuffer(e.target.files[0]);
};

(new THREE.STLLoader()).load('./example.stl', geometry => setGemotry(geometry));
